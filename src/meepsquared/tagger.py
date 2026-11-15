# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""A trained tagger, loaded for prediction.

    import meepsquared as m2
    tagger = m2.MeepTagger.from_pretrained("out/tagger")
    tagger.predict([["The", "hare", "runs", "fast", "."]])
    # [{"upos": [...], "lemma": [...], "head": [...], "deprel": [...]}]

The folder is what Result.save wrote for a TrainingJob(task="tagging"): the
adapter, the heads and their labels, and the tokenizer. The encoder is the
pretrained one the training named, loaded again, or the path given.
"""

from typing import Optional

from meepsquared import _tagging


class MeepTagger:
    """A tagger on its encoder, with its tokenizer: predict() labels sentences given as lists of words."""

    def __init__(self, tagger, tokenizer, config: dict):
        self.tagger, self.tokenizer, self.config = tagger, tokenizer, config

    @classmethod
    def from_pretrained(cls, path, encoder: Optional[str] = None, device: Optional[str] = None) -> "MeepTagger":
        """The tagger saved in `path`, on the encoder its training named, or on `encoder` (a model id or path of the
        same model, such as a local copy); on `device`, the first GPU when there is one."""
        import json
        from pathlib import Path

        import torch
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        config = json.loads((Path(path) / _tagging.CONFIG).read_text())
        name = encoder if encoder is not None else config["model"]
        device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        model_config = AutoConfig.from_pretrained(name)
        load_args = {"dtype": torch.bfloat16 if device.startswith("cuda") else torch.float32}
        if model_config.model_type in _tagging.BOTTLENECK_TYPES:
            load_args["add_pooling_layer"] = False
        loaded = AutoModel.from_pretrained(name, config=model_config, **load_args).to(device)
        tagger, config = _tagging.load(path, loaded, device)
        tagger.eval()
        return cls(tagger, AutoTokenizer.from_pretrained(path), config)

    def predict(self, sentences: list, decode: str = "mst", budget: int = 4096) -> list:
        """Each sentence's labels: for each tagging task a label per word (a lemma as the lemma), and with the parser
        "head" (from 1, 0 the root) and "deprel". decode: "mst", a tree with one root, or "argmax", each word's best
        head. budget: subwords per batch, rows times the longest."""
        rows = [{_tagging.TOKENS: list(words)} for words in sentences]
        limit = _tagging.max_subwords(self.tokenizer, self.tagger.encoder.config)
        # The words alone are encoded: prediction has no labels
        examples, dropped = _tagging.encode(self.tokenizer, rows, _tagging.Tasks({}, None), limit, "sentences")
        if dropped:
            raise ValueError(f"the sentences {dropped[:5]} have more than {limit} subwords, which the encoder takes")
        return _tagging.predict_examples(self.tagger, examples, rows, self.tokenizer.pad_token_id, budget, decode)

