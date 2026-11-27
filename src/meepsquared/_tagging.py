# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Token tagging and dependency parsing on a pretrained encoder (XLM-RoBERTa and its kind).

The data is sentences as words: a "tokens" column (the words of each
sentence) and, per word, the labels to learn:

- one column per tagging task, labels as text: "upos", "xpos", "feats",
  "lemma" or any other ("zsubj", "ner"); each becomes a head, a linear layer
  on the encoder's output for the first subword of each word;
- "head" (the number of each word's head, from 1, 0 for the root) and
  "deprel" (its relation): the parser, Dozat and Manning's biaffine
  scorer (2017): every word scores every other word and the root as its head,
  and each relation given its head.

A lemma is learned as an edit rule from the word ("case|strip|append", see
lemma_rule), so that one class covers every word inflected alike.

CoNLL-U files are read into those columns (read_conllu), as syntactic words
(UD's own unit; multi-word tokens left out) or as surface tokens (a
multi-word token as one token, see read_conllu).

Training sums the cross-entropy of every head and of the parser's heads and
relations. The encoder is frozen; what trains is the adapter (LoRA, or
bottleneck adapters after the feed-forward block of every layer, Pfeiffer et
al., 2020) and the heads. Prediction decodes each sentence's heads as a tree
with one root (Chu-Liu-Edmonds over the parser's scores), or as each word's
best head ("argmax", not always a tree).

What is saved is the adapter, the heads and the labels (save, load): the
encoder is the pretrained one, named in the saved folder.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch import nn

# The column of the words, and the parser's two
TOKENS, HEAD, DEPREL = "tokens", "head", "deprel"
# The task the parser is named by in Recipe(labels=[...])
PARSE = "parse"
# The tagging columns a CoNLL-U file gives, in its column order
CONLLU_TAGS = ("lemma", "upos", "xpos", "feats")
# The parser's widths: Dozat and Manning (2017) use 500 for heads and 100 for relations
ARC_DIM = 512
REL_DIM = 128
# A label of a word whose loss is not counted (PyTorch's cross-entropy ignore_index)
IGNORED = -100
WORDS = ("syntactic", "surface")
DECODES = ("mst", "argmax")
# The files of a saved tagger
WEIGHTS, CONFIG = "tagger.safetensors", "tagger.json"
# The model types whose layers have the feed-forward output projection bottleneck adapters follow
BOTTLENECK_TYPES = ("xlm-roberta", "xlm-roberta-xl", "roberta", "bert", "camembert")


# Lemmas as edit rules

def lemma_rule(form: str, lemma: str) -> str:
    """The rule that makes `lemma` from `form`: "case|strip|append".

    case "l": the lemma is lowercase and the rule applies to the lowercased
    form; "o": the rule applies to the form as written. strip: the characters
    removed from the end; append: the text added after.
    """
    case, base = ("l", form.lower()) if lemma == lemma.lower() else ("o", form)
    common = 0
    for a, b in zip(base, lemma):
        if a != b:
            break
        common += 1
    return f"{case}|{len(base) - common}|{lemma[common:]}"


def rule_strip(rule: str) -> int:
    return int(rule.split("|", 2)[1])


def apply_rule(form: str, rule: str) -> str:
    case, strip, append = rule.split("|", 2)
    base = form.lower() if case == "l" else form
    n = int(strip)
    if n > len(base):
        raise ValueError(f"lemma rule {rule!r} strips {n} characters from {form!r}")
    return base[:len(base) - n] + append


# Reading CoNLL-U

def _sentence(rows: list, where: str, words: str) -> dict:
    """One sentence's columns from its CoNLL-U rows, as syntactic words or surface tokens."""
    if words == "syntactic":
        kept = [r for r in rows if r[0].isdigit()]
        if [int(r[0]) for r in kept] != list(range(1, len(kept) + 1)):
            raise ValueError(f"{where}: word ids are not 1 to {len(kept)}")
        return {TOKENS: [r[1] for r in kept], "lemma": [r[2] for r in kept], "upos": [r[3] for r in kept],
                "xpos": [r[4] for r in kept], "feats": [r[5] for r in kept], HEAD: [int(r[6]) for r in kept],
                DEPREL: [r[7] for r in kept]}
    word_rows = {r[0]: r for r in rows if r[0].isdigit()}
    tokens = []  # (form, its word rows)
    i = 0
    while i < len(rows):
        row = rows[i]
        if "." in row[0]:
            i += 1
            continue
        if "-" in row[0]:
            first, last = (int(x) for x in row[0].split("-"))
            tokens.append((row[1], [word_rows[str(k)] for k in range(first, last + 1)]))
            i += 1
            while i < len(rows) and ("." in rows[i][0] or (rows[i][0].isdigit() and first <= int(rows[i][0]) <= last)):
                i += 1
            continue
        tokens.append((row[1], [row]))
        i += 1
    token_of = {"0": 0}
    for n, (_, parts) in enumerate(tokens, start=1):
        for part in parts:
            token_of[part[0]] = n
    out = {TOKENS: [], "lemma": [], "upos": [], "xpos": [], "feats": [], HEAD: [], DEPREL: []}
    for n, (form, parts) in enumerate(tokens, start=1):
        outside = [p for p in parts if token_of[p[6]] != n]
        if not outside:
            raise ValueError(f"{where}: token {n} {form!r} has no word with a head outside it")
        out[TOKENS].append(form)
        out["lemma"].append("+".join(p[2] for p in parts))
        out["upos"].append("+".join(p[3] for p in parts))
        out["xpos"].append("+".join(p[4] for p in parts))
        out["feats"].append(next((p[5] for p in parts if p[5] != "_"), "_"))
        out[HEAD].append(token_of[outside[0][6]])
        out[DEPREL].append(outside[0][7])
    return out


def read_conllu(path, words: str = "syntactic") -> list:
    """The sentences of a CoNLL-U file as rows of columns: tokens, lemma, upos, xpos, feats, head, deprel.

    words="syntactic": UD's words; multi-word token lines and empty nodes are
    left out. words="surface": the tokens as written; a multi-word token
    ("del" = de + el) is one token, its upos, xpos and lemma the parts' joined
    with "+", its feats the first part's that has any, its head and relation
    those of its first part whose head lies outside the token; a head that
    points into a multi-word token points to the token.
    """
    if words not in WORDS:
        raise ValueError(f"words must be one of {WORDS}, not {words!r}")
    out, rows, sent_id, line_number = [], [], None, 0
    path = Path(path)

    def flush():
        if rows:
            where = f"{path}, sentence {sent_id or len(out) + 1}"
            out.append(_sentence(rows, where, words))

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                flush()
                rows, sent_id = [], None
            elif line.startswith("# sent_id"):
                sent_id = line.split("=", 1)[1].strip()
            elif not line.startswith("#"):
                columns = line.split("\t")
                if len(columns) != 10:
                    raise ValueError(f"{path}:{line_number}: {len(columns)} columns, CoNLL-U has 10")
                rows.append(columns)
    flush()
    if not out:
        raise ValueError(f"{path}: no sentences")
    return out


# The tasks and their labels

@dataclass
class Tasks:
    """What a tagger learns: its tagging columns, each with its labels, and the parser's relations (None: no
    parser)."""

    tags: dict
    relations: Optional[list]

    def to_dict(self) -> dict:
        return {"tags": self.tags, "relations": self.relations}

    @classmethod
    def from_dict(cls, data: dict) -> "Tasks":
        return cls(dict(data["tags"]), data["relations"])

    def names(self) -> list:
        return list(self.tags) + ([PARSE] if self.relations is not None else [])


def tasks_of(columns: list, chosen: Optional[list]) -> list:
    """The tasks a tagger learns from data with these columns: `chosen`, or every tagging column and the parser
    when the data has head and deprel."""
    tags = [c for c in columns if c not in (TOKENS, HEAD, DEPREL)]
    available = tags + ([PARSE] if HEAD in columns and DEPREL in columns else [])
    if chosen is None:
        if not available:
            raise ValueError(f"the data's columns {columns} hold no labels: give tagging columns beside 'tokens', "
                             "or 'head' and 'deprel' for the parser")
        return available
    missing = [t for t in chosen if t not in available]
    if missing:
        raise ValueError(f"recipe.labels names {missing}, and the data has {available}"
                         + ("; 'parse' needs the columns 'head' and 'deprel'" if PARSE in missing else ""))
    return list(chosen)


def check_rows(rows, names: list, where: str) -> None:
    """Every sentence has words, and every chosen column one value per word."""
    columns = [n for n in names if n != PARSE] + ([HEAD, DEPREL] if PARSE in names else [])
    for i, row in enumerate(rows):
        n = len(row[TOKENS])
        if n == 0:
            raise ValueError(f"{where}, row {i}: a sentence without words")
        for column in columns:
            if len(row[column]) != n:
                raise ValueError(f"{where}, row {i}: {n} words and {len(row[column])} values of {column!r}")
        if PARSE in names and any(not 0 <= h <= n for h in row[HEAD]):
            raise ValueError(f"{where}, row {i}: a head outside 0 to {n}")


def label_of(column: str, form: str, value: str) -> str:
    return lemma_rule(form, value) if column == "lemma" else value


def build_tasks(rows, names: list) -> Tasks:
    """The labels of each task, as the training data has them, sorted."""
    tags = {}
    for name in names:
        if name == PARSE:
            continue
        tags[name] = sorted({label_of(name, form, value) for row in rows
                             for form, value in zip(row[TOKENS], row[name])})
    relations = sorted({d for row in rows for d in row[DEPREL]}) if PARSE in names else None
    return Tasks(tags, relations)


# Subwords

def max_subwords(tokenizer, config) -> int:
    """The most subwords of a sentence, the special tokens counted: the tokenizer's limit, which for
    XLM-RoBERTa's absolute positions is what the position table holds after the padding offset."""
    limit = tokenizer.model_max_length
    if limit is None or limit > 1_000_000:
        raise ValueError(f"the tokenizer of {config._name_or_path} states no length limit (model_max_length "
                         f"{limit}); a tagger needs it to know which sentences its encoder takes")
    return int(limit)


def encode(tokenizer, rows, tasks: Tasks, limit: int, where: str) -> tuple:
    """(examples, dropped): each sentence as input_ids, the position of each word's first subword, and the labels
    as indices (IGNORED where the training never saw a label); sentences over `limit` subwords are dropped and
    counted."""
    index = {name: {label: i for i, label in enumerate(labels)} for name, labels in tasks.tags.items()}
    relation = {label: i for i, label in enumerate(tasks.relations)} if tasks.relations is not None else None
    out, dropped = [], []
    for i, row in enumerate(rows):
        encoded = tokenizer(row[TOKENS], is_split_into_words=True, truncation=False)
        ids = encoded["input_ids"]
        if len(ids) > limit:
            dropped.append(i)
            continue
        first, seen = [], set()
        for position, word in enumerate(encoded.word_ids()):
            if word is not None and word not in seen:
                seen.add(word)
                first.append(position)
        if len(first) != len(row[TOKENS]):
            empty = [w for w in range(len(row[TOKENS])) if w not in seen]
            raise ValueError(f"{where}, row {i}: the tokenizer gives no subword to the words "
                             f"{[row[TOKENS][w] for w in empty]}")
        example = {"input_ids": ids, "first": first, "row": i}
        for name, labels in index.items():
            example[name] = [labels.get(label_of(name, form, value), IGNORED)
                             for form, value in zip(row[TOKENS], row[name])]
        if relation is not None:
            example[HEAD] = list(row[HEAD])
            example[DEPREL] = [relation.get(d, IGNORED) for d in row[DEPREL]]
        out.append(example)
    return out, dropped


class Collator:
    """Examples padded into a batch: input_ids with the pad token, attention_mask, `first` with -1, labels with
    IGNORED."""

    def __init__(self, pad_id: int, label_columns: list):
        self.pad_id = pad_id
        self.label_columns = label_columns

    def __call__(self, examples: list) -> dict:
        rows = len(examples)
        length = max(len(e["input_ids"]) for e in examples)
        words = max(len(e["first"]) for e in examples)
        batch = {"input_ids": torch.full((rows, length), self.pad_id, dtype=torch.long),
                 "attention_mask": torch.zeros((rows, length), dtype=torch.long),
                 "first": torch.full((rows, words), -1, dtype=torch.long)}
        for column in self.label_columns:
            batch[column] = torch.full((rows, words), IGNORED, dtype=torch.long)
        for b, e in enumerate(examples):
            n = len(e["input_ids"])
            batch["input_ids"][b, :n] = torch.tensor(e["input_ids"])
            batch["attention_mask"][b, :n] = 1
            batch["first"][b, :len(e["first"])] = torch.tensor(e["first"])
            for column in self.label_columns:
                batch[column][b, :len(e[column])] = torch.tensor(e[column])
        return batch


# Bottleneck adapters

class Bottleneck(nn.Module):
    """x + up(GELU(down(x))), up initialized to zero so that training starts from the encoder's own output."""

    def __init__(self, hidden: int, size: int):
        super().__init__()
        self.down = nn.Linear(hidden, size)
        self.up = nn.Linear(size, hidden)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return x + self.up(nn.functional.gelu(self.down(x)))


class _Adapted(nn.Module):
    """A layer's feed-forward output projection, then its bottleneck adapter. The adapter keeps its own dtype (float32)
    while the frozen encoder may be in 16 bits: the projection's output goes into the adapter in the adapter's dtype
    and comes back in its own, so the layer computes with or without autocast (the plan's timed steps run without
    it; under autocast the adapter's products run in 16 bits as the encoder's do)."""

    def __init__(self, dense: nn.Linear, adapter: Bottleneck):
        super().__init__()
        self.dense = dense
        self.adapter = adapter

    def forward(self, x):
        y = self.dense(x)
        return self.adapter(y.to(self.adapter.down.weight.dtype)).to(y.dtype)


def add_bottlenecks(encoder, size: int) -> int:
    """A bottleneck adapter after the feed-forward output projection of every layer (output.dense, Pfeiffer's
    place); returns their parameter count. Refused for a model type whose layers are not known to have it."""
    model_type = encoder.config.model_type
    if model_type not in BOTTLENECK_TYPES:
        raise ValueError(f"bottleneck adapters go after each layer's feed-forward output projection, known for "
                         f"{', '.join(BOTTLENECK_TYPES)}; this model is {model_type}: use method='lora'")
    count = 0
    for layer in encoder.encoder.layer:
        dense = layer.output.dense
        adapter = Bottleneck(dense.out_features, size).to(device=dense.weight.device)
        layer.output.dense = _Adapted(dense, adapter)
        count += sum(p.numel() for p in adapter.parameters())
    return count


# The parser and the tagger

def _mlp(hidden: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hidden, out), nn.GELU(), nn.Dropout(dropout))


class Parser(nn.Module):
    """Dozat and Manning's biaffine parser (2017): head scores for every word against every word and the root, and
    relation scores given a head."""

    def __init__(self, hidden: int, relations: int, dropout: float):
        super().__init__()
        self.root = nn.Parameter(torch.zeros(hidden))
        self.arc_dep, self.arc_head = _mlp(hidden, ARC_DIM, dropout), _mlp(hidden, ARC_DIM, dropout)
        self.rel_dep, self.rel_head = _mlp(hidden, REL_DIM, dropout), _mlp(hidden, REL_DIM, dropout)
        self.arc_w = nn.Parameter(torch.zeros(ARC_DIM, ARC_DIM))
        self.arc_b = nn.Linear(ARC_DIM, 1, bias=False)
        self.rel_w = nn.Parameter(torch.zeros(relations, REL_DIM + 1, REL_DIM + 1))

    def forward(self, words, count):
        """words (B, T, H) and each sentence's word count (B): head scores (B, T, T + 1), column 0 the root and
        columns past a sentence's end -inf, and the relation scorer, a function of a head per word (B, T)."""
        batch, length, _ = words.shape
        candidates = torch.cat([self.root.to(words.dtype).expand(batch, 1, -1), words], 1)
        dep, head = self.arc_dep(words), self.arc_head(candidates)
        arcs = torch.einsum("btd,de,bse->bts", dep, self.arc_w.to(dep.dtype), head) + self.arc_b(head).transpose(1, 2)
        valid = torch.arange(length + 1, device=words.device)[None, :] <= count[:, None]
        arcs = arcs.masked_fill(~valid[:, None, :], float("-inf"))
        rel_dep = nn.functional.pad(self.rel_dep(words), (0, 1), value=1.0)
        rel_head = nn.functional.pad(self.rel_head(candidates), (0, 1), value=1.0)

        def relations(heads):
            chosen = torch.gather(rel_head, 1, heads.clamp(min=0).unsqueeze(-1).expand(-1, -1, rel_head.size(-1)))
            return torch.einsum("btd,rde,bte->btr", rel_dep, self.rel_w.to(rel_dep.dtype), chosen)

        return arcs, relations


class Tagger(nn.Module):
    """A pretrained encoder with a head per tagging task and, with relations, the parser.

    forward(input_ids, attention_mask, first, <labels>) returns the scores of
    each task and, given labels, the loss: the sum of the tasks' cross-entropy,
    the parser's heads and relations counted as two.
    """

    # Transformers' Trainer passes loss arguments (num_items_in_batch) to a forward that takes **kwargs unless the
    # model says it does not; the tagger's keyword arguments are its labels, and its loss is a mean per batch
    accepts_loss_kwargs = False

    def __init__(self, encoder, tasks: Tasks, dropout: float):
        super().__init__()
        hidden = encoder.config.hidden_size
        self.encoder = encoder
        self.tasks = tasks
        self.drop = nn.Dropout(dropout)
        for name in tasks.tags:
            if not name.isidentifier():
                raise ValueError(f"a tagging column is named as a Python name (letters, digits, _), not {name!r}")
        self.heads = nn.ModuleDict({name: nn.Linear(hidden, len(labels)) for name, labels in tasks.tags.items()})
        self.parser = Parser(hidden, len(tasks.relations), dropout) if tasks.relations is not None else None

    def meepsquared_trunk(self):
        return self.encoder

    @property
    def label_columns(self) -> list:
        return list(self.tasks.tags) + ([HEAD, DEPREL] if self.parser is not None else [])

    def scores(self, input_ids, attention_mask, first) -> dict:
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        index = first.clamp(min=0).unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        words = self.drop(torch.gather(hidden, 1, index))
        out = {name: head(words) for name, head in self.heads.items()}
        if self.parser is not None:
            out["arcs"], out["relations"] = self.parser(words, (first >= 0).sum(1))
        return out

    def forward(self, input_ids, attention_mask, first, **labels):
        out = self.scores(input_ids, attention_mask, first)
        if not labels:
            return out
        unknown = sorted(set(labels) - set(self.label_columns))
        if unknown:
            raise ValueError(f"the tagger has no task for the columns {unknown}")
        loss = 0
        for name in self.tasks.tags:
            loss = loss + nn.functional.cross_entropy(out[name].float().flatten(0, 1), labels[name].flatten(),
                                                      ignore_index=IGNORED)
        if self.parser is not None:
            heads = labels[HEAD]
            loss = loss + nn.functional.cross_entropy(out["arcs"].float().flatten(0, 1), heads.flatten(),
                                                      ignore_index=IGNORED)
            loss = loss + nn.functional.cross_entropy(out["relations"](heads).float().flatten(0, 1),
                                                      labels[DEPREL].flatten(), ignore_index=IGNORED)
        out["loss"] = loss
        return out


# Decoding a tree

def _cycle(heads: np.ndarray) -> Optional[list]:
    """A cycle of the head array (node 0 is the root and has none), or None."""
    n = len(heads)
    done = np.zeros(n, dtype=bool)
    done[0] = True
    for start in range(1, n):
        path, node = [], start
        on_path = {}
        while not done[node] and node not in on_path:
            on_path[node] = len(path)
            path.append(node)
            node = heads[node]
        if not done[node] and node in on_path:
            return path[on_path[node]:]
        done[path] = True
    return None


def chu_liu_edmonds(scores: np.ndarray) -> np.ndarray:
    """The highest-scoring tree: scores[d, h] is head h for word d, node 0 the root; returns each node's head, -1
    for the root."""
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
    outside = np.flatnonzero(~in_cycle)  # node 0 first
    inside = np.array(cycle)
    m = len(outside)
    contracted = np.full((m + 1, m + 1), -np.inf)
    contracted[:m, :m] = s[np.ix_(outside, outside)]
    # The cycle as a head: its best member for each outside word
    out_of = s[np.ix_(outside, inside)]
    contracted[:m, m] = out_of.max(1)
    best_out = inside[out_of.argmax(1)]
    # The cycle as a word: the best way in from each outside head, against the cycle's own edge it replaces
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
    """The highest-scoring tree in which one word only has the root as its head (UD's rule): the best tree, or,
    when it has several, the best of the trees with each candidate word alone under the root."""
    heads = chu_liu_edmonds(scores)
    roots = np.flatnonzero(heads[1:] == 0) + 1
    if len(roots) <= 1:
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
    """One sentence's heads from its head scores (T, T + 1): a tree with one root ("mst"), or each word's best
    head ("argmax")."""
    if decode == "argmax":
        return arcs[:count, :count + 1].argmax(-1).tolist()
    logp = torch.log_softmax(arcs[:count, :count + 1].float(), -1).cpu().numpy()
    scores = np.full((count + 1, count + 1), -np.inf)
    scores[1:, :] = logp
    return one_root_tree(scores)[1:].tolist()


# Prediction and scores

def batches_by_length(lengths: list, budget: int) -> list:
    """Example indices in order of length, grouped while rows x longest stays within `budget`."""
    out, group, longest = [], [], 0
    for i in sorted(range(len(lengths)), key=lengths.__getitem__):
        if group and max(longest, lengths[i]) * (len(group) + 1) > budget:
            out.append(group)
            group, longest = [], 0
        group.append(i)
        longest = max(longest, lengths[i])
    if group:
        out.append(group)
    return out


@torch.no_grad()
def predict_examples(tagger: Tagger, examples: list, rows: list, pad_id: int, budget: int, decode: str) -> list:
    """Each example's predictions, in the order of `examples`: a label per word for each tagging task (a lemma as
    the lemma itself), and with the parser "head" and "deprel"."""
    if decode not in DECODES:
        raise ValueError(f"decode must be one of {DECODES}, not {decode!r}")
    was_training = tagger.training
    tagger.eval()
    device = next(tagger.parameters()).device
    tasks = tagger.tasks
    collate = Collator(pad_id, [])
    strip = None
    if "lemma" in tasks.tags:
        strip = torch.tensor([rule_strip(r) for r in tasks.tags["lemma"]], device=device)
    out = [None] * len(examples)
    try:
        for group in batches_by_length([len(e["input_ids"]) for e in examples], budget):
            batch = {k: v.to(device) for k, v in collate([examples[i] for i in group]).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                scores = tagger.scores(batch["input_ids"], batch["attention_mask"], batch["first"])
            if tagger.parser is not None:
                # Every sentence's tree first, then the relations of the whole batch given those heads
                heads = torch.zeros_like(batch["first"])
                for b, i in enumerate(group):
                    n = len(rows[examples[i]["row"]][TOKENS])
                    heads[b, :n] = torch.tensor(decode_heads(scores["arcs"][b], n, decode), device=device)
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    relations = scores["relations"](heads).float().argmax(-1)
            for b, i in enumerate(group):
                row = rows[examples[i]["row"]]
                forms = row[TOKENS]
                n = len(forms)
                found = {}
                for name, labels in tasks.tags.items():
                    task_scores = scores[name][b, :n].float()
                    if name == "lemma":
                        lengths = torch.tensor([len(f) for f in forms], device=device)
                        task_scores = task_scores.masked_fill(strip[None, :] > lengths[:, None], -math.inf)
                        found[name] = [apply_rule(f, labels[k]) for f, k in zip(forms, task_scores.argmax(-1).tolist())]
                    else:
                        found[name] = [labels[k] for k in task_scores.argmax(-1).tolist()]
                if tagger.parser is not None:
                    found[HEAD] = heads[b, :n].tolist()
                    found[DEPREL] = [tasks.relations[k] for k in relations[b, :n].tolist()]
                out[i] = found
    finally:
        tagger.train(was_training)
    return out


def scores_of(predictions: list, examples: list, rows: list, tasks: Tasks) -> dict:
    """Accuracy of each tagging task over all words, and with the parser UAS (the right head) and LAS (the right
    head and relation); a label the training never saw counts as wrong."""
    right, total = {name: 0 for name in tasks.tags}, 0
    uas = las = 0
    for found, example in zip(predictions, examples):
        row = rows[example["row"]]
        total += len(row[TOKENS])
        for name in tasks.tags:
            right[name] += sum(p == g for p, g in zip(found[name], row[name]))
        if tasks.relations is not None:
            for ph, pd, gh, gd in zip(found[HEAD], found[DEPREL], row[HEAD], row[DEPREL]):
                uas += ph == gh
                las += ph == gh and pd == gd
    result = {name: right[name] / total for name in tasks.tags}
    if tasks.relations is not None:
        result["uas"], result["las"] = uas / total, las / total
    result["words"] = total
    return result


def score(scores: dict, tasks: Tasks) -> float:
    """One number to keep the best epoch by: the mean of each tagging task's accuracy and, with the parser, LAS."""
    parts = [scores[name] for name in tasks.tags] + ([scores["las"]] if tasks.relations is not None else [])
    return sum(parts) / len(parts)


# Saving and loading

def trainable_state(tagger) -> dict:
    """The adapter and the heads: every parameter that trains, under its name."""
    return {name: p.detach() for name, p in tagger.named_parameters() if p.requires_grad}


def save(tagger, path, config: dict, state: Optional[dict] = None) -> None:
    """The adapter and the heads (WEIGHTS), and what rebuilds the tagger around its encoder (CONFIG)."""
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = state if state is not None else trainable_state(tagger)
    save_file({k: v.detach().float().contiguous().cpu() for k, v in state.items()}, str(path / WEIGHTS))
    (path / CONFIG).write_text(json.dumps({**config, "tasks": tagger.tasks.to_dict()}, ensure_ascii=False, indent=1))


def build(encoder, config: dict, tasks: Tasks):
    """A tagger on `encoder` with the adapter `config` names (method "bottleneck" or "lora") and the heads of
    `tasks`, its encoder frozen."""
    for p in encoder.parameters():
        p.requires_grad = False
    method = config["method"]
    if method == "bottleneck":
        add_bottlenecks(encoder, config["bottleneck"])
    elif method == "lora":
        from peft import LoraConfig, get_peft_model

        encoder = get_peft_model(encoder, LoraConfig(r=config["rank"], lora_alpha=config["alpha"],
                                                     lora_dropout=config["lora_dropout"],
                                                     target_modules=config["targets"]))
    else:
        raise ValueError(f"a tagger's method is 'bottleneck' or 'lora', not {method!r}")
    tagger = Tagger(encoder, tasks, config["dropout"])
    tagger.to(next(encoder.parameters()).device)
    return tagger


def load(path, encoder, device=None):
    """The tagger saved in `path`, on `encoder` (the pretrained encoder its config names, loaded by the caller)."""
    from safetensors.torch import load_file

    path = Path(path)
    config = json.loads((path / CONFIG).read_text())
    tagger = build(encoder, config, Tasks.from_dict(config["tasks"]))
    state = load_file(str(path / WEIGHTS), device=str(device) if device is not None else "cpu")
    names = {name for name, p in tagger.named_parameters() if p.requires_grad}
    if set(state) != names:
        raise ValueError(f"{path}: the saved weights are not this tagger's: missing {sorted(names - set(state))[:5]}, "
                         f"unknown {sorted(set(state) - names)[:5]}")
    parameters = dict(tagger.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            parameters[name].copy_(value.to(parameters[name].dtype))
    return tagger, config
