# Plan: a choice-model package with serving, and the Japanese subject finder

Status: agreed with the person on 2026-11-24. Two sessions do the work:

- **kubexlat**: the Japanese subject finder. Owns tkamd2 for its trainings,
  the data, the answer key of 201 predicates, and the 31B teacher.
- **apps-eb**: Meep², Forge and Thinkube Control. Owns tkspark for its
  checks and test trainings, and builds the serving package.

Rule for both: one GPU training per node at a time (heat). The 31B on tkspark
is loaded and unloaded by kubexlat when needed; no other task uses it, so
apps-eb has tkspark whenever the 31B is unloaded.

## The goals

1. A Japanese subject finder that runs on one 24 GB GPU, as close as possible
   to the teacher (Gemma 4 31B with checks: 188 of 201).
2. A package: training in Meep² (the choice task) and a separate serving
   package that serves a base for generation and the choice head together.
3. Evidence: fair comparisons with the other open choice-model projects.

## What is measured so far (plans on the 3090, data 9bee827b)

| Model | GPU peak | Per step | Full epoch | One milestone |
|---|---|---|---|---|
| Gemma 4 12B, NF4, 4 candidates a pass | 21.75 GiB | 17.3 s | 33.7 h | 6 min |
| Gemma 4 26B-A4B, NF4 experts, 2 a pass | 22.86 GiB | 26 s | 51.4 h | 12 min |
| Gemma 4 E4B, NF4 (old data) | 14.03 GiB | 9.2 s | 16.5 h | 2 min |

The E4B student reached 156 of 201 and went flat after about 7,600 questions.
The new candidate builder reaches all 201 (190 before).

## Step 0: decisions (the person)

- The main model: the 12B is recommended (more room, faster, no expert
  workaround).

## Step 1: three tracks at once

| Track | Who | Node | Work |
|---|---|---|---|
| A | kubexlat | tkamd2 | Train the main model on data 9bee827b (key b60a9568), a milestone about every 1,500 questions; stop when the count on the 201 goes flat. Report each milestone. |
| B | apps-eb builds, then runs | tkspark | Head options in the choice task: (1) a separate "none applies" output before the choice among candidates; (2) a second interaction layer over the candidates. Then short E4B trainings on data 9bee827b, stopped at about 7,600 questions: the current head, and each option. Train them in FP8 or NVFP4, with the milestones answered on the NF4 base (milestone_weights). Each run is also the first whole-run check of FP8, NVFP4 and milestone_weights. kubexlat reads the milestone answers and compares them on the 201. |
| C | kubexlat | CPU | The Kyoto University Wikipedia Annotated Corpus in the question format: our tagger, our candidate builder, the annotated ガ argument as the right candidate (著者, 読者, 不特定:人 as なし); its reach; articles of the 201 left out. Calibration of the milestones' answers (Brier score, accuracy by confidence). |

Also apps-eb, while B runs: Forge's field for milestone_weights; the plan
fitting the memory itself on out of memory; neutral names in Meep²'s code and
plan texts (the choice task's own words, the credit kept in NOTICE only).

## Step 2: serving (apps-eb builds, kubexlat tests)

- The serving package around the E4B student that exists (its milestone with
  156, Forge run c30b4cd1), on tkamd2's second 3090 when no training needs
  the node's attention: the choice head alone first.
- Then one base for generation and the choice head together
  (design/serving-generation-and-choice.md); measure 4-bit generation speed.
- Then routing: the head's confident answers kept, the rest answered by
  generation (the base, or the 31B, loaded by kubexlat).
- kubexlat scores each serving setup on the 201 and measures time per
  predicate (0.64 s for the E4B in the milestones).

## Step 3: the final Japanese model (kubexlat)

Train the best combination from Steps 1 and 2 (size, head options, corpus
data) on tkamd2; serve it with the package on a 3090; score it on the 201 as
served.

## Step 4: evidence, then release

- apps-eb: train an open choice model's setup with Meep² on its public data
  and score it on System1-Arena against the published models.
- kubexlat: train the open models (OpenJev-4B, Open-Jev-2B) on our data and
  score them on the 201.
- apps-eb: documentation with the measured numbers; a release (0.1.x) when the
  person asks for it.
