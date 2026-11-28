# meepserve

Trained parts on a base model, served behind Thinkube's LLM gateway: a part is
a folder of trained weights (adapters, heads) for a base, with a contract file
`meepserve.json` naming its kind, format and base. The design is in
`design/serving-package.md` of this repository.

```python
import meepserve

part = meepserve.load("parts/ja-parse", "models/xlm-roberta-large")
part.answer({"sentences": [["今日", "は", "晴れ"]], "probabilities": ["upos"]})
```

Kinds are found through the Python entry point group `meepserve.kinds`; this
package has the kind `tagger` (bottleneck adapters, heads and a biaffine
parser on an encoder, or a splitter of text into words) and the kind `choice`
(a LoRA adapter and a head on a causal language model, choosing among
candidates, as Meep²'s choice task saves it; `pip install "meepserve[choice]"`). It depends on
PyTorch and Transformers only; Meep² trains parts with its modules, and it
never imports Meep².

Licence: Apache-2.0.
