# The serving package: trained parts on a base, behind the LLM gateway

Status: agreed with the person on 2026-11-24; the core and the tagger kind are
built first.

## The need

Two kinds of trained models must run behind the LLM gateway, called by Mill
tasks, with no model loaded inside a task:

- KubeXlat's Japanese analysis: a splitter, a tagger and parser, and a
  detector of left-out ガ, ヲ and ニ, each a set of bottleneck adapters and
  heads on FacebookAI/xlm-roberta-large.
- The choice model: a LoRA adapter and the set head on a Gemma 4 base, and
  generation from the same base without a second copy
  (serving-generation-and-choice.md).

Both are the same thing: a base model, and small trained parts read from saved
files. Serving needs only the code that runs those parts; it does not need the
training library, and the training library must not depend on serving.

## The design

A package of its own, `meepserve` (folder `serve/` of this repository, its own
pyproject and dependencies: PyTorch and Transformers; vLLM only for the vLLM
engine). It holds the code that runs trained parts. Meep² imports the parts'
modules from it to train them; it never imports Meep².

### Kinds

A kind is one sort of trained part. It declares:

- its saved format: a folder whose config names the kind, its format version
  and the base model, beside its weight files; a format version the package
  does not know is refused, never guessed;
- its modules (heads, a parser, the set head);
- what it reads from the base: a tagger, the state of each word's first piece;
  the choice head, the last state of each candidate after a shared text;
- its request and its answer.

Kinds are found through the Python entry point group `meepserve.kinds`: a
package registers a kind, so a project's own kind can live in its own
repository, and neither the core nor the gateway changes for it. The package
registers its own kinds the same way.

First kinds:

- `tagger`: bottleneck adapters after the feed-forward block of every encoder
  layer, a linear head per label set, and optionally Dozat and Manning's
  biaffine parser. Its format 1 is the one KubeXlat's taggers were saved in
  (tagger.json and weights.pt), so ja-seg, ja-parse and ja-zero are served as
  they are. Its input is words (each word's first piece) or, for a splitter
  (its config's input "one character per piece"), one vocabulary piece per
  character with a run of spaces as "▁", in windows of 500 pieces that overlap
  by 100.
- `choice` (next): the LoRA adapter and the set head of Meep²'s choice task,
  in the format Meep² saves (adapter_model.safetensors, head.safetensors,
  choice.json).

### Engines

An engine holds the base and gives a kind the states it asks for, with the
kind's adapter applied. The kind does not know which engine runs it.

- `transformers`: the base with Transformers and PyTorch, one request batch at
  a time; adapters inserted in the base's modules, switchable by name, so
  several parts can share one loaded base.
- `vllm` (with the choice kind): the base held once for generation and for the
  choice head's states (serving-generation-and-choice.md).

### The gateway

- One route for every kind: `POST /v1/heads`, a JSON body naming the model;
  the rest of the body is the kind's request. The gateway's proxy forwards it
  to the pod that serves the model, as it forwards embeddings.
- A trained part is a model of the catalogue: task `heads`, server_type
  `transformers` (or the engine's backend), and `base_model`, the catalogue id
  of its base. Its files are a model version in MLflow, as any model's.
- The transformers backend serves task `heads`: it loads the base from its
  MLflow mirror and the part from its own model version, and answers
  `POST /v1/heads` with the part's kind.

### Training

Meep² trains a kind with the kind's modules from this package, and adds what
only training needs: the data, the loss, the schedule. It saves in the kind's
format, through the package, so what Forge trains is what the gateway serves.
Meep²'s tagging task moves onto the package's tagger modules when a tagger is
next trained.

## The tagger kind's request and answer

A splitter (labels `seg`, B and I):

    {"model": "...", "texts": ["今日は晴れ。", ...]}
    -> {"words": [["今日", "は", "晴れ", "。"], ...]}

Any other tagger:

    {"model": "...", "sentences": [["今日", "は", ...], ...],
     "probabilities": ["ga", "wo", "ni"], "decode": "argmax"}
    -> {"sentences": [{"upos": [...], "lemma": [...], "head": [...],
                       "deprel": [...], "probabilities": {"ga": [{"0": p, "1": p}, ...]}}, ...]}

- One label per word for each head (a lemma as the lemma its rule makes); with
  the parser, `head` (from 1, 0 the root) and `deprel`.
- `probabilities` names the heads whose label probabilities come back, per
  word; a head with thousands of labels (lemma rules) is asked for only by
  name.
- `decode`: `argmax`, each word's best head, as the taggers were evaluated;
  or `mst`, a tree with one root (Chu-Liu-Edmonds).
- A sentence longer than the encoder takes (510 pieces) is refused with its
  index.

## Order of work

1. This note.
2. The package's core (kinds, the transformers engine) and the tagger kind,
   with tests on a small random encoder.
3. The gateway: the `heads` route in the proxy, task `heads` in the
   transformers backend, and registration of a trained part in the catalogue.
4. ja-seg, ja-parse and ja-zero in MLflow and the catalogue; checked by their
   own test scores (test.json) through the gateway.
5. The choice kind and the vLLM engine (the serving plan's Step 2).
