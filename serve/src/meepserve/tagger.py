# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""The tagger kind: bottleneck adapters and heads on an encoder, labelling words or splitting text into words.

Format 1, the files KubeXlat's taggers were saved in:

- tagger.json: "lang" (the adapters' name), "bottleneck" (their size),
  "labels" ({head: [label, ...]}, "deprel" for the parser's relations),
  "encoder" (the base), and for a splitter "input": "one character per
  piece";
- weights.pt: the adapters, "adapters.<layer>.down|up.weight|bias", and the
  heads, "heads.out.<head>.weight|bias" and "heads.parser.<name>"; read with
  torch.load(weights_only=True), every name required and none other allowed.

A head is a linear layer on the encoder's state of each word's first piece.
The parser is Dozat and Manning's biaffine one (2017): every word scores every
other word and the root as its head, and each relation given its head.

A splitter has the one head "seg", labels B (begins a word) and I. Its text
goes to the encoder one character per piece: each character is looked up in
the encoder's vocabulary as a piece on its own (unknown when it is not one),
and a run of spaces is the piece "▁"; a space belongs to no word and the
character after it begins one. A text of more than MAX_PIECES pieces is read
in windows of WINDOW pieces that overlap by OVERLAP, each character taking
the label of the window in which it is farthest from an edge.

Requests (see answer): {"texts": [...]} to a splitter, {"sentences": [[word,
...], ...], "probabilities": [head, ...], "decode": "argmax" or "mst"} to any
other tagger.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from meepserve import CONTRACT, PartError

FORMATS = (1,)
SPLITTER_INPUT = "one character per piece"
SEG, SEG_LABELS = "seg", ("B", "I")
LEMMA, DEPREL = "lemma", "deprel"
# The pieces of one reading besides the two special ones, as the taggers were trained
MAX_PIECES = 510
WINDOW, OVERLAP = 500, 100
SPACE_PIECE = "▁"
# Pieces per batch: rows times the longest
BUDGET = 4096
DECODES = ("argmax", "mst")
# The parser's widths
ARC_DIM, REL_DIM = 512, 128


def _mlp(hidden: int, out: int) -> nn.Sequential:
    # The dropout of training is kept as a module so that the saved names (index 0, the linear layer) match
    return nn.Sequential(nn.Linear(hidden, out), nn.GELU(), nn.Dropout(0.0))


class Parser(nn.Module):
    def __init__(self, hidden: int, relations: int):
        super().__init__()
        self.root = nn.Parameter(torch.zeros(hidden))
        self.arc_dep, self.arc_head = _mlp(hidden, ARC_DIM), _mlp(hidden, ARC_DIM)
        self.rel_dep, self.rel_head = _mlp(hidden, REL_DIM), _mlp(hidden, REL_DIM)
        self.arc_w = nn.Parameter(torch.zeros(ARC_DIM, ARC_DIM))
        self.arc_b = nn.Linear(ARC_DIM, 1, bias=False)
        self.rel_w = nn.Parameter(torch.zeros(relations, REL_DIM + 1, REL_DIM + 1))

    def forward(self, words, count):
        """words (B, T, H), each sentence's word count (B): head scores (B, T, T + 1), column 0 the root and the
        columns past a sentence's end -inf, and the relation scorer, a function of a head per word (B, T)."""
        batch, length, _ = words.shape
        candidates = torch.cat([self.root.expand(batch, 1, -1), words], 1)
        dep, head = self.arc_dep(words), self.arc_head(candidates)
        arcs = torch.einsum("btd,de,bse->bts", dep, self.arc_w, head) + self.arc_b(head).transpose(1, 2)
        valid = torch.arange(length + 1, device=words.device)[None, :] <= count[:, None]
        arcs = arcs.masked_fill(~valid[:, None, :], float("-inf"))
        rel_dep = nn.functional.pad(self.rel_dep(words), (0, 1), value=1.0)
        rel_head = nn.functional.pad(self.rel_head(candidates), (0, 1), value=1.0)

        def relations(heads):
            chosen = torch.gather(rel_head, 1, heads.clamp(min=0).unsqueeze(-1).expand(-1, -1, rel_head.size(-1)))
            return torch.einsum("btd,rde,bte->btr", rel_dep, self.rel_w, chosen)

        return arcs, relations


class Heads(nn.Module):
    def __init__(self, hidden: int, labels: dict):
        super().__init__()
        self.out = nn.ModuleDict({name: nn.Linear(hidden, len(values)) for name, values in labels.items()
                                  if name != DEPREL})
        self.parser = Parser(hidden, len(labels[DEPREL])) if DEPREL in labels else None

    def forward(self, words, count) -> dict:
        out = {name: head(words) for name, head in self.out.items()}
        if self.parser is not None:
            out["arcs"], out["relations"] = self.parser(words, count)
        return out


# Lemmas as edit rules, "case|strip|append": case "l", the rule applies to the lowercased form (and the lemma is
# lowercase), "o" to the form as written; strip characters are removed from its end and append is added

def rule_strip(rule: str) -> int:
    return int(rule.split("|", 2)[1])


def apply_rule(form: str, rule: str) -> str:
    case, strip, append = rule.split("|", 2)
    base = form.lower() if case == "l" else form
    return base[:len(base) - int(strip)] + append


# A tree from the parser's scores (Chu-Liu-Edmonds)

def _cycle(heads: np.ndarray):
    n = len(heads)
    done = np.zeros(n, dtype=bool)
    done[0] = True
    for start in range(1, n):
        path, node, on_path = [], start, {}
        while not done[node] and node not in on_path:
            on_path[node] = len(path)
            path.append(node)
            node = heads[node]
        if not done[node] and node in on_path:
            return path[on_path[node]:]
        done[path] = True
    return None


def chu_liu_edmonds(scores: np.ndarray) -> np.ndarray:
    """The highest-scoring tree: scores[d, h] is head h for word d, node 0 the root; each node's head, -1 for the
    root."""
    n = scores.shape[0]
    s = scores.astype(np.float64).copy()
    np.fill_diagonal(s, -np.inf)
    s[0, :] = -np.inf
    heads = s.argmax(1)
    heads[0] = -1
    cycle = _cycle(heads)
    if cycle is None:
        return heads
    in_cycle = np.zeros(n, dtype=bool)
    in_cycle[cycle] = True
    outside = np.flatnonzero(~in_cycle)
    inside = np.array(cycle)
    m = len(outside)
    contracted = np.full((m + 1, m + 1), -np.inf)
    contracted[:m, :m] = s[np.ix_(outside, outside)]
    out_of = s[np.ix_(outside, inside)]
    contracted[:m, m] = out_of.max(1)
    best_out = inside[out_of.argmax(1)]
    kept = s[inside, heads[inside]]
    into = s[np.ix_(inside, outside)] - kept[:, None]
    contracted[m, :m] = into.max(0)
    best_in = inside[into.argmax(0)]
    tree = chu_liu_edmonds(contracted)
    out = heads.copy()
    for i in range(1, m):
        h = tree[i]
        out[outside[i]] = outside[h] if h < m else best_out[i]
    h = tree[m]
    out[best_in[h]] = outside[h]
    return out


def one_root_tree(scores: np.ndarray) -> np.ndarray:
    """The highest-scoring tree in which one word only has the root as its head."""
    heads = chu_liu_edmonds(scores)
    if len(np.flatnonzero(heads[1:] == 0)) <= 1:
        return heads
    best, best_score = None, -np.inf
    words = np.arange(1, scores.shape[0])
    for root in words[np.isfinite(scores[1:, 0])]:
        s = scores.copy()
        s[:, 0] = -np.inf
        s[root, 0] = scores[root, 0]
        tree = chu_liu_edmonds(s)
        total = s[words, tree[1:]].sum()
        if total > best_score:
            best, best_score = tree, total
    return best


def decode_heads(arcs: torch.Tensor, count: int, decode: str) -> list:
    """One sentence's heads (from 1, 0 the root) from its head scores (T, T + 1)."""
    if decode == "argmax":
        return arcs[:count, :count + 1].argmax(-1).tolist()
    logp = torch.log_softmax(arcs[:count, :count + 1].float(), -1).cpu().numpy()
    scores = np.full((count + 1, count + 1), -np.inf)
    scores[1:, :] = logp
    return one_root_tree(scores)[1:].tolist()


# Reading the encoder

@dataclass
class _Reading:
    """One reading of the encoder: its pieces, and the position of each labelled unit's piece."""
    ids: list
    first: list


def _batches(readings: list, budget: int) -> list:
    """Indexes of the readings in batches of at most `budget` pieces, rows times the longest, in order."""
    batches, current, longest = [], [], 0
    for i, r in enumerate(readings):
        n = len(r.ids)
        if current and max(longest, n) * (len(current) + 1) > budget:
            batches.append(current)
            current, longest = [], 0
        current.append(i)
        longest = max(longest, n)
    if current:
        batches.append(current)
    return batches


def _collate(readings: list, pad_id: int, device):
    width = max(len(r.ids) for r in readings)
    words = max(len(r.first) for r in readings)
    ids = torch.full((len(readings), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(readings), width), dtype=torch.long)
    first = torch.full((len(readings), words), -1, dtype=torch.long)
    for b, r in enumerate(readings):
        ids[b, :len(r.ids)] = torch.tensor(r.ids)
        mask[b, :len(r.ids)] = 1
        first[b, :len(r.first)] = torch.tensor(r.first)
    return ids.to(device), mask.to(device), first.to(device)


class TaggerPart:
    """A tagger on its engine: answer(request) labels words, or splits texts into words for a splitter."""

    def __init__(self, name: str, labels: dict, splitter: bool, heads: Heads, engine):
        self.name, self.labels, self.splitter, self.heads, self.engine = name, labels, splitter, heads, engine
        self.vocab = engine.tokenizer.get_vocab()

    def _outputs(self, readings: list):
        """Per batch: (indexes, outputs of the heads, first) for the readings, the adapters of this part active."""
        tokenizer = self.engine.tokenizer
        for batch in _batches(readings, BUDGET):
            ids, mask, first = _collate([readings[i] for i in batch], tokenizer.pad_token_id, self.engine.device)
            with self.engine.reading(self.name):
                hidden = self.engine.states(ids, mask)
                index = first.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
                out = self.heads(torch.gather(hidden, 1, index), (first >= 0).sum(1))
                yield batch, out, first

    # A splitter

    def _pieces(self, text: str) -> tuple:
        """Piece ids and, for each piece, its character's index in text (-1 for a space piece)."""
        ids, where, k = [], [], 0
        unknown = self.engine.tokenizer.unk_token_id
        while k < len(text):
            if text[k].isspace():
                while k < len(text) and text[k].isspace():
                    k += 1
                ids.append(self.vocab[SPACE_PIECE])
                where.append(-1)
                continue
            ids.append(self.vocab.get(text[k], unknown))
            where.append(k)
            k += 1
        return ids, where

    def split(self, texts: list) -> list:
        """The words of each text, in order."""
        tokenizer = self.engine.tokenizer
        jobs, readings = [], []  # jobs: (text index, the characters a reading labels)
        for t, text in enumerate(texts):
            ids, where = self._pieces(text)
            windows = ([(s, s + WINDOW) for s in range(0, len(ids) - OVERLAP, WINDOW - OVERLAP)]
                       if len(ids) > MAX_PIECES else [(0, len(ids))])
            for s, e in windows:
                chars = [w for w in where[s:e] if w >= 0]
                if not chars:
                    continue
                jobs.append((t, chars))
                readings.append(_Reading([tokenizer.cls_token_id] + ids[s:e] + [tokenizer.sep_token_id],
                                         [i + 1 for i, w in enumerate(where[s:e]) if w >= 0]))
        best = [dict() for _ in texts]  # character index -> (distance from the window's edge, label)
        for batch, out, _ in self._outputs(readings):
            labels = out[SEG].argmax(-1).tolist()
            for b, i in enumerate(batch):
                t, chars = jobs[i]
                for k, ch in enumerate(chars):
                    edge = min(k, len(chars) - 1 - k)
                    if ch not in best[t] or edge > best[t][ch][0]:
                        best[t][ch] = (edge, SEG_LABELS[labels[b][k]])
        words = []
        for t, text in enumerate(texts):
            spans, start = [], None
            for pos, ch in enumerate(text):
                if ch.isspace():
                    if start is not None:
                        spans.append((start, pos))
                        start = None
                    continue
                if best[t][pos][1] == "B" or start is None:
                    if start is not None:
                        spans.append((start, pos))
                    start = pos
            if start is not None:
                spans.append((start, len(text)))
            words.append([text[a:b] for a, b in spans])
        return words

    # Any other tagger

    def _encode(self, sentences: list) -> list:
        tokenizer = self.engine.tokenizer
        readings = []
        for i, words in enumerate(sentences):
            if not words:
                readings.append(None)
                continue
            encoded = tokenizer(words, is_split_into_words=True, truncation=False)
            ids = encoded["input_ids"]
            if len(ids) > MAX_PIECES + 2:
                raise PartError(f"sentence {i} has {len(ids) - 2} pieces, more than the {MAX_PIECES} the tagger "
                                "reads")
            first, seen = [], set()
            for position, word in enumerate(encoded.word_ids()):
                if word is not None and word not in seen:
                    seen.add(word)
                    first.append(position)
            if len(first) != len(words):
                raise PartError(f"sentence {i}: {len(words)} words make pieces for {len(first)}; a word is empty")
            readings.append(_Reading(ids, first))
        return readings

    def tag(self, sentences: list, probabilities: list, decode: str) -> list:
        """Each sentence's labels: a label per word for each head (a lemma as the lemma its rule makes), the
        parser's head and deprel, and the probabilities of the heads named."""
        found = [None] * len(sentences)
        for i, words in enumerate(sentences):
            if not words:
                found[i] = self._empty(probabilities)
        readings = self._encode(sentences)
        kept = [i for i, r in enumerate(readings) if r is not None]
        rules = self.labels.get(LEMMA)
        strips = (torch.tensor([rule_strip(r) for r in rules], device=self.engine.device)
                  if rules is not None else None)
        for batch, out, first in self._outputs([readings[i] for i in kept]):
            heads_of = None
            if self.heads.parser is not None:
                arcs = out["arcs"].float()
                heads_of = [decode_heads(arcs[b], len(readings[kept[i]].first), decode) for b, i in enumerate(batch)]
                width = arcs.size(1)
                chosen = torch.tensor([h + [0] * (width - len(h)) for h in heads_of], device=arcs.device)
                with torch.no_grad(), torch.autocast(arcs.device.type, dtype=torch.bfloat16,
                                                     enabled=arcs.device.type == "cuda"):
                    relations = out["relations"](chosen).argmax(-1).tolist()
            for b, i in enumerate(batch):
                index = kept[i]
                words = sentences[index]
                n = len(words)
                answer = {}
                for name in self.heads.out:
                    scores = out[name][b, :n].float()
                    if name == LEMMA:
                        lengths = torch.tensor([len(w) for w in words], device=scores.device)
                        scores = scores.masked_fill(strips[None, :] > lengths[:, None], -math.inf)
                        answer[name] = [apply_rule(w, rules[k]) for w, k in zip(words, scores.argmax(-1).tolist())]
                    else:
                        answer[name] = [self.labels[name][k] for k in scores.argmax(-1).tolist()]
                if heads_of is not None:
                    answer["head"] = heads_of[b]
                    answer[DEPREL] = [self.labels[DEPREL][k] for k in relations[b][:n]]
                if probabilities:
                    answer["probabilities"] = {
                        name: [dict(zip(self.labels[name], p)) for p in out[name][b, :n].float().softmax(-1).tolist()]
                        for name in probabilities}
                found[index] = answer
        return found

    def _empty(self, probabilities: list) -> dict:
        answer = {name: [] for name in self.heads.out}
        if self.heads.parser is not None:
            answer.update({"head": [], DEPREL: []})
        if probabilities:
            answer["probabilities"] = {name: [] for name in probabilities}
        return answer

    def answer(self, request: dict) -> dict:
        """{"texts": [...]} -> {"words": [[...], ...]} for a splitter; {"sentences": [[...], ...], "probabilities":
        [head, ...], "decode": "argmax" (default) or "mst"} -> {"sentences": [{...}, ...]} for any other tagger."""
        given = {k: v for k, v in request.items() if k != "model"}
        if self.splitter:
            unknown = sorted(set(given) - {"texts"})
            if unknown:
                raise PartError(f"a splitter takes 'texts' only, not {unknown}")
            texts = given.get("texts")
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                raise PartError("'texts' must be a list of texts")
            return {"words": self.split(texts)}
        unknown = sorted(set(given) - {"sentences", "probabilities", "decode"})
        if unknown:
            raise PartError(f"a tagger takes 'sentences', 'probabilities' and 'decode', not {unknown}")
        sentences = given.get("sentences")
        if not isinstance(sentences, list) or not all(
                isinstance(s, list) and all(isinstance(w, str) and w for w in s) for s in sentences):
            raise PartError("'sentences' must be a list of sentences, each a list of non-empty words")
        probabilities = given.get("probabilities") or []
        if not isinstance(probabilities, list) or not all(p in self.heads.out for p in probabilities):
            raise PartError(f"'probabilities' names heads of this tagger: {sorted(self.heads.out)}")
        decode = given.get("decode", "argmax")
        if decode not in DECODES:
            raise PartError(f"'decode' is one of {list(DECODES)}, not {decode!r}")
        if decode == "mst" and self.heads.parser is None:
            raise PartError("'decode' is for a tagger with a parser; this one has none")
        return {"sentences": self.tag(sentences, probabilities, decode)}


def _split_state(state: dict) -> tuple:
    adapters = {k[len("adapters."):]: v for k, v in state.items() if k.startswith("adapters.")}
    heads = {k[len("heads."):]: v for k, v in state.items() if k.startswith("heads.")}
    other = sorted(set(state) - {f"adapters.{k}" for k in adapters} - {f"heads.{k}" for k in heads})
    if other:
        raise PartError(f"weights.pt holds names that are neither adapters nor heads: {other[:4]}")
    return adapters, heads


def load(folder: Path, contract: dict, engine) -> TaggerPart:
    """A tagger part of format 1 from `folder`, its adapters added to `engine`."""
    config = json.loads((folder / "tagger.json").read_text())
    for field in ("lang", "bottleneck", "labels"):
        if field not in config:
            raise PartError(f"{folder}/tagger.json has no '{field}'")
    if config.get("encoder") is not None and config["encoder"] != contract["base"]:
        raise PartError(f"{folder}: tagger.json names the encoder {config['encoder']}, and {CONTRACT} the base "
                        f"{contract['base']}")
    labels = config["labels"]
    splitter = config.get("input") == SPLITTER_INPUT
    if splitter and (set(labels) != {SEG} or tuple(labels[SEG]) != SEG_LABELS):
        raise PartError(f"{folder}: a splitter has the one head {SEG}, labels {list(SEG_LABELS)}")
    state = torch.load(folder / "weights.pt", map_location="cpu", weights_only=True)
    adapters, head_state = _split_state(state)
    heads = Heads(engine.hidden, labels)
    missing, unknown = heads.load_state_dict(head_state, strict=False)
    if missing or unknown:
        raise PartError(f"{folder}: the heads in weights.pt do not fit its labels: missing {missing[:4]}, unknown "
                        f"{unknown[:4]}")
    # The adapters are kept in the engine under the tagger's own name: parts that share a base differ by it
    name = config["lang"]
    engine.add_bottlenecks(name, int(config["bottleneck"]), adapters)
    return TaggerPart(name, labels, splitter, heads.to(engine.device).eval(), engine)



@dataclass(frozen=True)
class _Kind:
    name: str
    formats: tuple
    engine: str

    def load(self, folder: Path, contract: dict, engine) -> TaggerPart:
        return load(folder, contract, engine)


KIND = _Kind("tagger", FORMATS, "transformers")
