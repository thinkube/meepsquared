# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0
# From the zebra-grpo recipe of Thinkube's Unsloth notebooks, by the same authors, relicensed.
"""
zebra_dataset.py — Generate zebra-puzzle training data for GRPO.

Each puzzle is guaranteed to have a unique solution (verified by a CSP solver).
Clue sets are minimised: removing any clue produces multiple solutions.

Clue types match the ZebraLogic benchmark (Lin et al., 2024):
  Found_At, Not_At, Same_House, Direct_Left/Right, Side_By_Side,
  Left/Right_Of, One_Between, Two_Between

JSONL row schema:
  prompt:     natural-language puzzle the model sees
  expected:   ground-truth one-string answer (for SFT warm-up)
  solution:   structured ground truth, used by reward()
  categories: dict cat -> list of values
  N, theme, num_clues

Usage:
  python zebra_dataset.py --demo                    # print one puzzle
  python zebra_dataset.py --n 1000 --out data.jsonl
"""
from __future__ import annotations
import argparse, json, random, re
from typing import Callable, Dict, List, Optional

from constraint import Problem, AllDifferentConstraint

# ---------------------------------------------------------------------------
# Themes — add more for variety. All categories must have N values.
# ---------------------------------------------------------------------------
THEMES = [
    {
        "name": "classic",
        "categories": {
            "color":       ["red", "green", "ivory", "yellow", "blue"],
            "nationality": ["Englishman", "Spaniard", "Ukrainian", "Norwegian", "Japanese"],
            "drink":       ["coffee", "tea", "milk", "orange juice", "water"],
            "pet":         ["dog", "snail", "fox", "horse", "zebra"],
            "smoke":       ["Old Gold", "Kools", "Chesterfields", "Lucky Strike", "Parliaments"],
        },
    },
    {
        "name": "music_school",
        "categories": {
            "instrument": ["piano", "violin", "drums", "guitar", "flute"],
            "student":    ["Alex", "Blair", "Casey", "Drew", "Emma"],
            "lesson_day": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
            "genre":      ["jazz", "classical", "rock", "blues", "folk"],
            "snack":      ["apples", "cookies", "chips", "yogurt", "nuts"],
        },
    },
    {
        "name": "tapas_bar",
        "categories": {
            "tapa":   ["patatas bravas", "croquetas", "boquerones", "tortilla", "pulpo"],
            "wine":   ["rioja", "albariño", "verdejo", "cava", "tempranillo"],
            "region": ["Galicia", "Catalonia", "Andalusia", "Basque Country", "La Rioja"],
            "diner":  ["Marta", "Joan", "Núria", "Pep", "Laia"],
            "time":   ["19:00", "20:00", "21:00", "22:00", "23:00"],
        },
    },
]

# ---------------------------------------------------------------------------
# CSP plumbing
# ---------------------------------------------------------------------------
def vname(cat: str, val: str) -> str:
    return f"{cat}::{val}"

def base_problem(cats: Dict[str, List[str]], N: int) -> Problem:
    p = Problem()
    for c, vs in cats.items():
        for v in vs:
            p.addVariable(vname(c, v), list(range(N)))
        p.addConstraint(AllDifferentConstraint(), [vname(c, v) for v in vs])
    return p

def n_solutions(p: Problem, cap: int = 2) -> int:
    n = 0
    for _ in p.getSolutionIter():
        n += 1
        if n >= cap:
            return n
    return n

def random_solution(cats, rng) -> Dict[str, List[str]]:
    return {c: rng.sample(vs, len(vs)) for c, vs in cats.items()}

def pos_of(sol, c, v): return sol[c].index(v)
def val_at(sol, c, p): return sol[c][p]

# Per-category phrasing for natural-sounding clues.
# Default is "the {val}" — override only when that reads badly.
PHRASING = {
    "student":    "{val}",
    "diner":      "{val}",
    "lesson_day": "the lesson on {val}",
    "time":       "the diner at {val}",
    "region":     "the diner from {val}",
}
def phrase(cat: str, val: str) -> str:
    return PHRASING.get(cat, "the {val}").format(val=val)

# ---------------------------------------------------------------------------
# Clue generators — all ten ZebraLogic clue types
# ---------------------------------------------------------------------------
def cl_position(sol, cats, N, rng):
    cat = rng.choice(list(cats.keys()))
    val = rng.choice(cats[cat])
    pos = pos_of(sol, cat, val)
    loc = ("the middle house" if (N == 5 and pos == 2)
           else "the first house" if pos == 0
           else "the last house" if pos == N - 1
           else f"house {pos + 1}")
    np = phrase(cat, val)
    text = (np[0].upper() + np[1:]) + f" is in {loc}."
    return {"text": text,
            "fn": (lambda tp: (lambda a: a == tp))(pos),
            "vars": [vname(cat, val)]}

def cl_same(sol, cats, N, rng):
    c1, c2 = rng.sample(list(cats.keys()), 2)
    p = rng.randrange(N)
    v1, v2 = val_at(sol, c1, p), val_at(sol, c2, p)
    np1 = phrase(c1, v1)
    text = (np1[0].upper() + np1[1:]) + f" is in the same house as {phrase(c2, v2)}."
    return {"text": text,
            "fn": (lambda a, b: a == b),
            "vars": [vname(c1, v1), vname(c2, v2)]}

def cl_immediate_left(sol, cats, N, rng):
    p1 = rng.randrange(N - 1); p2 = p1 + 1
    c1 = rng.choice(list(cats.keys())); c2 = rng.choice(list(cats.keys()))
    v1, v2 = val_at(sol, c1, p1), val_at(sol, c2, p2)
    if c1 == c2 and v1 == v2: return None
    np1 = phrase(c1, v1)
    text = (np1[0].upper() + np1[1:]) + f" is immediately to the left of {phrase(c2, v2)}."
    return {"text": text,
            "fn": (lambda a, b: a == b - 1),
            "vars": [vname(c1, v1), vname(c2, v2)]}

def cl_immediate_right(sol, cats, N, rng):
    p2 = rng.randrange(N - 1); p1 = p2 + 1
    c1 = rng.choice(list(cats.keys())); c2 = rng.choice(list(cats.keys()))
    v1, v2 = val_at(sol, c1, p1), val_at(sol, c2, p2)
    if c1 == c2 and v1 == v2: return None
    np1 = phrase(c1, v1)
    text = (np1[0].upper() + np1[1:]) + f" is immediately to the right of {phrase(c2, v2)}."
    return {"text": text,
            "fn": (lambda a, b: a == b + 1),
            "vars": [vname(c1, v1), vname(c2, v2)]}

def cl_next_to(sol, cats, N, rng):
    p1 = rng.randrange(N)
    p2 = p1 + rng.choice([-1, 1])
    if not (0 <= p2 < N): return None
    c1 = rng.choice(list(cats.keys())); c2 = rng.choice(list(cats.keys()))
    v1, v2 = val_at(sol, c1, p1), val_at(sol, c2, p2)
    if c1 == c2 and v1 == v2: return None
    np1 = phrase(c1, v1)
    text = (np1[0].upper() + np1[1:]) + f" is next to {phrase(c2, v2)}."
    return {"text": text,
            "fn": (lambda a, b: abs(a - b) == 1),
            "vars": [vname(c1, v1), vname(c2, v2)]}

def cl_somewhere_left(sol, cats, N, rng):
    c1 = rng.choice(list(cats.keys())); c2 = rng.choice(list(cats.keys()))
    p1 = rng.randrange(N - 1)
    p2 = rng.randrange(p1 + 1, N)
    v1, v2 = val_at(sol, c1, p1), val_at(sol, c2, p2)
    if c1 == c2 and v1 == v2: return None
    np1 = phrase(c1, v1)
    text = (np1[0].upper() + np1[1:]) + f" is somewhere to the left of {phrase(c2, v2)}."
    return {"text": text,
            "fn": (lambda a, b: a < b),
            "vars": [vname(c1, v1), vname(c2, v2)]}

def cl_somewhere_right(sol, cats, N, rng):
    c1 = rng.choice(list(cats.keys())); c2 = rng.choice(list(cats.keys()))
    p2 = rng.randrange(N - 1)
    p1 = rng.randrange(p2 + 1, N)
    v1, v2 = val_at(sol, c1, p1), val_at(sol, c2, p2)
    if c1 == c2 and v1 == v2: return None
    np1 = phrase(c1, v1)
    text = (np1[0].upper() + np1[1:]) + f" is somewhere to the right of {phrase(c2, v2)}."
    return {"text": text,
            "fn": (lambda a, b: a > b),
            "vars": [vname(c1, v1), vname(c2, v2)]}

def cl_not_at(sol, cats, N, rng):
    cat = rng.choice(list(cats.keys()))
    val = rng.choice(cats[cat])
    true_pos = pos_of(sol, cat, val)
    wrong_positions = [i for i in range(N) if i != true_pos]
    pos = rng.choice(wrong_positions)
    loc = ("the middle house" if (N == 5 and pos == 2)
           else "the first house" if pos == 0
           else "the last house" if pos == N - 1
           else f"house {pos + 1}")
    np = phrase(cat, val)
    text = (np[0].upper() + np[1:]) + f" is not in {loc}."
    return {"text": text,
            "fn": (lambda wp: (lambda a: a != wp))(pos),
            "vars": [vname(cat, val)]}

def cl_one_between(sol, cats, N, rng):
    p1 = rng.randrange(N)
    p2 = p1 + 2 * rng.choice([-1, 1])
    if not (0 <= p2 < N): return None
    c1 = rng.choice(list(cats.keys())); c2 = rng.choice(list(cats.keys()))
    v1, v2 = val_at(sol, c1, p1), val_at(sol, c2, p2)
    if c1 == c2 and v1 == v2: return None
    return {"text": f"There is one house between {phrase(c1, v1)} and {phrase(c2, v2)}.",
            "fn": (lambda a, b: abs(a - b) == 2),
            "vars": [vname(c1, v1), vname(c2, v2)]}

def cl_two_between(sol, cats, N, rng):
    p1 = rng.randrange(N)
    p2 = p1 + 3 * rng.choice([-1, 1])
    if not (0 <= p2 < N): return None
    c1 = rng.choice(list(cats.keys())); c2 = rng.choice(list(cats.keys()))
    v1, v2 = val_at(sol, c1, p1), val_at(sol, c2, p2)
    if c1 == c2 and v1 == v2: return None
    return {"text": f"There are two houses between {phrase(c1, v1)} and {phrase(c2, v2)}.",
            "fn": (lambda a, b: abs(a - b) == 3),
            "vars": [vname(c1, v1), vname(c2, v2)]}

CLUE_GENS: List[Callable] = [cl_position, cl_same, cl_immediate_left,
                              cl_immediate_right, cl_next_to,
                              cl_somewhere_left, cl_somewhere_right,
                              cl_not_at, cl_one_between, cl_two_between]

# ---------------------------------------------------------------------------
# Puzzle generation
# ---------------------------------------------------------------------------
def build(cats, N, clues) -> Problem:
    p = base_problem(cats, N)
    for c in clues:
        p.addConstraint(c["fn"], c["vars"])
    return p

def minimise(clues, cats, N):
    i = 0
    while i < len(clues):
        trial = clues[:i] + clues[i + 1:]
        if n_solutions(build(cats, N, trial)) == 1:
            clues = trial
        else:
            i += 1
    return clues

def generate(theme, N, rng, max_clues=25, max_attempts=600) -> Optional[dict]:
    cats = theme["categories"]
    sol = random_solution(cats, rng)
    clues, seen, tries = [], set(), 0
    while tries < max_attempts and len(clues) < max_clues:
        tries += 1
        clue = rng.choice(CLUE_GENS)(sol, cats, N, rng)
        if clue is None or clue["text"] in seen:
            continue
        clues.append(clue); seen.add(clue["text"])
        n = n_solutions(build(cats, N, clues))
        if n == 0:
            clues.pop(); seen.discard(clue["text"])
        elif n == 1:
            clues = minimise(clues, cats, N)
            return {"theme": theme["name"], "N": N,
                    "categories": {k: list(v) for k, v in cats.items()},
                    "clues": [c["text"] for c in clues], "solution": sol}
    return None

# ---------------------------------------------------------------------------
# Prompt, expected-answer formatting, and reward
# ---------------------------------------------------------------------------
def format_prompt(p: dict) -> str:
    cats, N = p["categories"], p["N"]
    out = [f"There are {N} houses in a row, numbered 1 to {N} from left to right.",
           "Each house has exactly one of each attribute:"]
    for c, vs in cats.items():
        out.append(f"  - {c}: {', '.join(vs)}")
    out += ["", "Clues:"]
    out += [f"  {i}. {c}" for i, c in enumerate(p["clues"], 1)]
    out += ["",
            "Reason step by step, then write the final answer with one line per house:",
            "House 1: " + " | ".join(f"{c}=<value>" for c in cats),
            "(and similarly for houses 2 to {}).".format(N)]
    return "\n".join(out)

def format_expected(p: dict) -> str:
    cats = list(p["categories"].keys()); sol = p["solution"]
    return "\n".join(
        f"House {i + 1}: " + " | ".join(f"{c}={sol[c][i]}" for c in cats)
        for i in range(p["N"]))

_HOUSE_RE = re.compile(r"^\s*house\s+(\d+)\s*:\s*(.+?)\s*$", re.I)

def parse_answer(text: str, cats: List[str], N: int) -> Dict[str, List[Optional[str]]]:
    out = {c: [None] * N for c in cats}
    low_cats = {c.lower(): c for c in cats}
    for line in text.splitlines():
        m = _HOUSE_RE.match(line)
        if not m: continue
        i = int(m.group(1)) - 1
        if not (0 <= i < N): continue
        for part in re.split(r"[|,;]", m.group(2)):
            if "=" not in part: continue
            k, v = part.split("=", 1)
            real = low_cats.get(k.strip().lower())
            if real: out[real][i] = v.strip()
    return out

def reward(generated: str, puzzle: dict) -> float:
    """Continuous reward in [0, 1.25]: fraction-correct + 0.25 bonus if perfect."""
    cats = list(puzzle["categories"].keys())
    parsed = parse_answer(generated, cats, puzzle["N"])
    correct = total = 0
    for c in cats:
        for i in range(puzzle["N"]):
            total += 1
            got = parsed[c][i]
            want = puzzle["solution"][c][i]
            if got and got.lower() == want.lower():
                correct += 1
    return correct / total + (0.25 if correct == total else 0.0)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--N", type=int, default=5)
    ap.add_argument("--out", default="zebra_dataset.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    if args.demo:
        theme = rng.choice(THEMES)
        p = None
        while p is None:
            p = generate(theme, args.N, rng)
        print("=== PROMPT ===")
        print(format_prompt(p))
        print("\n=== EXPECTED ANSWER ===")
        print(format_expected(p))
        print(f"\n=== META ===\ntheme={p['theme']}  clues={len(p['clues'])}")
        return

    with open(args.out, "w", encoding="utf-8") as f:
        written = 0
        while written < args.n:
            theme = rng.choice(THEMES)
            p = generate(theme, args.N, rng)
            if p is None:
                continue
            row = {
                "prompt": format_prompt(p),
                "expected": format_expected(p),
                "solution": p["solution"],
                "categories": p["categories"],
                "N": p["N"], "theme": p["theme"], "num_clues": len(p["clues"]),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
            if written % 25 == 0:
                print(f"... {written}/{args.n}")
        print(f"wrote {written} puzzles to {args.out}")

if __name__ == "__main__":
    main()
