# Copyright Alejandro Martínez Corriá and the Thinkube contributors
# SPDX-License-Identifier: Apache-2.0
# From the zebra-grpo recipe of Thinkube's Unsloth notebooks, by the same authors, relicensed.

"""The zebra task of the GRPO benchmark: the model translates each clue of a
zebra puzzle into a formal rule, a constraint solver deduces the grid from the
rules, and the reward is the fraction of the grid's cells the solver gets right."""

import json
import random
import re

from constraint import AllDifferentConstraint, Problem

import zebra_dataset

def make_puzzles(num_puzzles, seed):
    """Generate puzzles with a fixed RNG."""
    rng = random.Random(seed)
    puzzles = []
    while len(puzzles) < num_puzzles:
        theme = rng.choice(zebra_dataset.THEMES)
        puzzle = zebra_dataset.generate(theme, 5, rng)
        if puzzle is None:
            continue
        puzzles.append(puzzle)
    return puzzles


RULE_FORMS = """{"type":"at","item":["category","value"],"house":2}
{"type":"not_at","item":["category","value"],"house":2}
{"type":"same","a":["category","value"],"b":["category","value"]}
{"type":"immediate_left","a":["category","value"],"b":["category","value"]}
{"type":"immediate_right","a":["category","value"],"b":["category","value"]}
{"type":"next_to","a":["category","value"],"b":["category","value"]}
{"type":"left_of","a":["category","value"],"b":["category","value"]}
{"type":"right_of","a":["category","value"],"b":["category","value"]}
{"type":"one_between","a":["category","value"],"b":["category","value"]}
{"type":"two_between","a":["category","value"],"b":["category","value"]}"""


def translation_prompt(puzzle):
    cats_block = "\n".join(f"  {c}: {', '.join(vs)}" for c, vs in puzzle["categories"].items())
    clues_block = "\n".join(f"  {i}. {c}" for i, c in enumerate(puzzle["clues"], 1))
    return (
        "Translate each zebra-puzzle clue into one formal rule.\n\n"
        f"Categories and values:\n{cats_block}\n\n"
        f"Houses are numbered 1 to {puzzle['N']} from left to right.\n"
        "'immediate_left' means a is directly left of b; 'left_of' means a is anywhere left of b.\n"
        "'one_between' means exactly one house stands between a and b.\n\n"
        f"Rule forms (JSON, one per clue):\n{RULE_FORMS}\n\n"
        "Answer with ONLY a JSON array containing one rule per clue, in order. "
        "Copy category and value names exactly as given.\n\n"
        f"Clues:\n{clues_block}"
    )


def parse_rules(text):
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return None
    try:
        rules = json.loads(m.group(0))
        return rules if isinstance(rules, list) else None
    except Exception:
        return None


_PAIR_CONSTRAINTS = {
    "same": lambda a, b: a == b,
    "immediate_left": lambda a, b: a == b - 1,
    "immediate_right": lambda a, b: a == b + 1,
    "next_to": lambda a, b: abs(a - b) == 1,
    "left_of": lambda a, b: a < b,
    "right_of": lambda a, b: a > b,
    "one_between": lambda a, b: abs(a - b) == 2,
    "two_between": lambda a, b: abs(a - b) == 3,
}


def solve_rules(rules, puzzle):
    """Apply translated rules with a CSP solver.

    Returns (grid_or_None, n_valid, n_invalid). The rules come from model
    output and can have any malformed shape: anything that does not match
    the schema exactly counts as invalid and is skipped. Contradictory rule
    sets return None (infeasible).
    """
    cats, N = puzzle["categories"], puzzle["N"]
    p = Problem()
    for c, vs in cats.items():
        for v in vs:
            p.addVariable(f"{c}::{v}", list(range(N)))
        p.addConstraint(AllDifferentConstraint(), [f"{c}::{v}" for v in vs])

    def valid_pair(x):
        return (isinstance(x, (list, tuple)) and len(x) == 2
                and isinstance(x[0], str) and isinstance(x[1], str)
                and x[0] in cats and x[1] in cats[x[0]])

    n_valid = n_invalid = 0
    for r in rules:
        try:
            t = r.get("type") if isinstance(r, dict) else None
            if t in ("at", "not_at"):
                item, house = r.get("item"), r.get("house")
                if valid_pair(item) and isinstance(house, int) and 1 <= house <= N:
                    pos = house - 1
                    fn = (lambda tp: lambda a: a == tp) if t == "at" else (lambda tp: lambda a: a != tp)
                    p.addConstraint(fn(pos), [f"{item[0]}::{item[1]}"])
                    n_valid += 1
                else:
                    n_invalid += 1
            elif t in _PAIR_CONSTRAINTS:
                a, b = r.get("a"), r.get("b")
                if valid_pair(a) and valid_pair(b) and (a[0] != b[0] or a[1] != b[1]):
                    p.addConstraint(_PAIR_CONSTRAINTS[t], [f"{a[0]}::{a[1]}", f"{b[0]}::{b[1]}"])
                    n_valid += 1
                else:
                    n_invalid += 1
            else:
                n_invalid += 1
        except Exception:
            n_invalid += 1

    try:
        sol = p.getSolution()
    except Exception:
        return None, n_valid, n_invalid
    if not sol:
        return None, n_valid, n_invalid
    grid = {c: [None] * N for c in cats}
    for key, pos in sol.items():
        c, v = key.split("::", 1)
        grid[c][pos] = v
    return grid, n_valid, n_invalid


def grade_grid(grid, puzzle):
    """Fraction of grid cells matching the true solution (0.0 for no grid)."""
    if grid is None:
        return 0.0
    sol, cats, N = puzzle["solution"], list(puzzle["categories"].keys()), puzzle["N"]
    hits = sum(1 for c in cats for i in range(N)
               if grid[c][i] and grid[c][i].lower() == sol[c][i].lower())
    return hits / (len(cats) * N)


# Training dataset: the prompt asks for rules; the reward re-derives the grid
def puzzle_to_row(puzzle):
    return {
        "prompt": [{"role": "user", "content": translation_prompt(puzzle)}],
        "puzzle_solution": json.dumps(puzzle["solution"]),
        "puzzle_categories": json.dumps(puzzle["categories"]),
        "puzzle_n": puzzle["N"],
        "num_clues": len(puzzle["clues"]),
    }


def zebra_reward_func(completions, puzzle_solution, puzzle_categories,
                      puzzle_n, num_clues, **kwargs):
    """GRPO reward: solver-graded cell accuracy of the translated rules.

    - unparseable output: 0
    - parseable: fraction of correct grid cells (0..1)
    - +0.05 format bonus for one rule per clue
    - +0.25 bonus for a fully correct grid
    """
    rewards = []
    for completion, sol_json, cats_json, n_val, n_clues in zip(
        completions, puzzle_solution, puzzle_categories, puzzle_n, num_clues
    ):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        puzzle = {
            "solution": json.loads(sol_json),
            "categories": json.loads(cats_json),
            "N": int(n_val),
        }
        rules = parse_rules(text)
        if rules is None:
            rewards.append(0.0)
            continue
        grid, n_valid, n_invalid = solve_rules(rules, puzzle)
        score = grade_grid(grid, puzzle)
        if len(rules) == int(n_clues):
            score += 0.05
        if grid is not None and grade_grid(grid, puzzle) == 1.0:
            score += 0.25
        rewards.append(float(score))
    return rewards
