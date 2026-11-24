# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0

"""Trained parts on a base model, served: a part's folder, its kind, and the engine that runs its base.

A part is a folder of trained weights for a base model (adapters, heads),
with a contract file, meepserve.json:

    {"kind": "tagger", "format": 1, "base": "FacebookAI/xlm-roberta-large"}

kind names the kind that reads and runs it (see kinds()); format, the
version of the kind's files, which the kind must know; base, the catalogue id
of the base model. The base's own files are given apart, as a folder.

    import meepserve
    part = meepserve.load("parts/ja-parse", "models/xlm-roberta-large")
    part.answer({"sentences": [["今日", "は", "晴れ"]]})

A kind is found through the Python entry point group "meepserve.kinds": its
object has name, formats, engine (the engine's name) and load(folder,
contract, engine), which gives the part; a part has answer(request), a
request and an answer being JSON objects, and close(), which removes what it
added to its engine. Several parts on one base share one engine when the same
engine is given to each load; the engine holds the base until it is dropped.
"""

import json
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional

CONTRACT = "meepserve.json"
ENTRY_POINTS = "meepserve.kinds"


class PartError(ValueError):
    """A part cannot be loaded, or a request cannot be answered; the message says why."""


def kinds() -> dict:
    """{name: kind} of every kind installed, through the entry point group meepserve.kinds."""
    found = {}
    for point in entry_points(group=ENTRY_POINTS):
        kind = point.load()
        if kind.name != point.name:
            raise PartError(f"the entry point {point.name} of {ENTRY_POINTS} gives a kind named {kind.name!r}")
        found[kind.name] = kind
    return found


def read_contract(folder) -> dict:
    """The part's meepserve.json, checked: kind, format and base are required."""
    path = Path(folder) / CONTRACT
    if not path.is_file():
        raise PartError(f"{folder} has no {CONTRACT}: a part's folder says its kind, format and base there")
    contract = json.loads(path.read_text())
    for field, kind in (("kind", str), ("format", int), ("base", str)):
        if not isinstance(contract.get(field), kind) or isinstance(contract.get(field), bool):
            raise PartError(f"{path}: '{field}' is required, a {kind.__name__}")
    return contract


def engine_for(contract: dict, base, device: Optional[str] = None):
    """A new engine of the kind's for the base folder `base`."""
    kind = _kind(contract)
    if kind.engine == "transformers":
        from meepserve.engine import TransformersEncoder

        return TransformersEncoder(base, device)
    raise PartError(f"the kind {kind.name} runs on the engine {kind.engine!r}, which this package does not have")


def load(folder, base=None, device: Optional[str] = None, engine=None):
    """The part in `folder`, on the base model in the folder `base` or on `engine`, an engine already holding that
    base (to share it between parts)."""
    contract = read_contract(folder)
    kind = _kind(contract)
    if contract["format"] not in kind.formats:
        raise PartError(f"{folder}: format {contract['format']} of the kind {kind.name} is not known to this "
                        f"package, which reads formats {list(kind.formats)}")
    if engine is None:
        if base is None:
            raise PartError("give the base model's folder, or an engine that holds it")
        engine = engine_for(contract, base, device)
    return kind.load(Path(folder), contract, engine)


def _kind(contract: dict):
    found = kinds()
    if contract["kind"] not in found:
        raise PartError(f"no kind {contract['kind']!r} is installed; installed: {sorted(found)}")
    return found[contract["kind"]]
