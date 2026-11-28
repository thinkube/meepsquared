# Plan: a tool-routing showcase of a choice head and an LLM on one base

Status: planned, not started. For an article about Meep² and meepserve; it
names no platform, and its code runs with Meep², meepserve, Hugging Face
datasets and one 24 GB GPU.

## What it shows

One copy of a base model does two things for an agent:

1. **The choice.** Given the user's request and the tools available, a
   choice head (Meep²'s choice task: a LoRA adapter and the set head) picks
   the tool, or "no tool" when none fits or none is needed. It answers in
   one pass, the shared text read once and each tool read after it, so its
   cost grows little with the number of tools, and it gives a probability
   for each tool.
2. **The generation.** The same base writes the call's arguments as JSON,
   for the chosen tool only: the prompt carries one tool's schema, not all
   of them.

Against the same base doing ordinary function calling (one generation that
picks the tool and writes its arguments, every tool in the prompt).

## What the article measures

- The tool chosen right.
- "No tool" said when it should be.
- Calibration: how well the head's probability matches how often it is
  right, so that a router can ask instead of acting when it is unsure.
- The arguments right, by BFCL's syntax-tree check.
- Time and prompt tokens per request, against function calling with every
  tool in the prompt; the saving grows with the number of tools.

## Data

The exact names, sizes and formats are checked on Hugging Face when the
work starts.

- Training: xLAM function-calling 60k (Salesforce APIGen: about 60,000
  requests with their tools and calls over about 3,600 APIs), ToolACE for
  more tools per request, and requests that need no tool for "no tool".
- Test: BFCL (arguments, and its irrelevance cases), MetaTool (choosing the
  tool, and whether one is needed), and the routing tasks of S1-Arena as a
  test outside the training data.

## Base

Gemma 4 E4B, which fits a 24 GB GPU in 16 bits; Gemma 4 12B in 4-bit NF4 as
the larger point.

## The example

`examples/tool-routing/` in this repository, plain scripts with no platform
in them:

| Script | What it does |
|---|---|
| `prepare.py` | xLAM, ToolACE, BFCL and MetaTool as choice questions: the tools and "no tool" as candidates, the right ones marked |
| `train.py` | Meep²'s choice task on the base, with milestones on a held-out slice |
| `serve.py` | meepserve's dual inference: the choice head and generation from one copy of the base |
| `compare.py` | the measures above, against plain function calling |

Where these run on a cluster with its own services for data work and
training, those services run the same scripts' functions; nothing in the
scripts depends on them.

## Order of work

1. `prepare.py`, its datasets checked first.
2. `train.py` on Gemma 4 E4B.
3. `serve.py`, once meepserve serves the choice head and generation from one
   copy of the base (serving-generation-and-choice.md, and the vLLM
   hidden-state path being tested for it).
4. `compare.py`, and the figures for the article.
