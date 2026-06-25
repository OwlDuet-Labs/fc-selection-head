# Across-BFCL-categories selector experiments

This closes **Paper 1 Limitation (c)** ("single task family — the trained-vs-vanilla
result is shown only on `live_multiple`"). It extends the BFCL data builder to three
more BFCL categories and adds a runner that produces one across-category comparison
table, so the headline claim ("a trained 0.6B selector matches a vanilla model 7× its
size") can be reported across the BFCL task families, not just `live_multiple`.

## Categories

| category | rows | shape | gold | what it tests |
|---|---:|---|---|---|
| `live_multiple` | 1053 | N candidate functions, gold among them | named (possible_answer) | discrimination (the paper's headline set) |
| `live_simple` | 258 | exactly 1 candidate (K=1) | named (possible_answer) | the easy floor of the difficulty axis |
| `live_relevance` | 16 | 1–5 functions, ≥1 applies | **none named** | should CALL a function (don't over-abstain) |
| `live_irrelevance` | 884 | 0–N functions, none applies | **none named** | should ABSTAIN (don't hallucinate a call) |

`live_multiple` and `live_simple` carry a `possible_answer/BFCL_v4_<cat>.json` file, so
gold is read exactly as before (first key of `ground_truth[0]`). `live_relevance` and
`live_irrelevance` have **no** possible-answer file and **no** per-row named gold — BFCL
scores them structurally (did the model call something / abstain).

## Abstention convention (load-bearing)

The selector head emits a **1-based index**, so "no tool applies" needs an index to point
at. The builder therefore appends an explicit sentinel candidate **`NONE`**
("No function applies — abstain (do not call any function).") as the **last** candidate of
every relevance/irrelevance instance. Then:

- **`live_irrelevance`** (no function applies) → `gold = "NONE"`, `gold_position = K` (the
  appended sentinel), `abstain = true`. Exact-match against `gold_position` already scores
  this as "abstain-correct".
- **`live_relevance`** (some function applies) → there is no single named gold index, so by
  convention `gold = candidates[0]`, `gold_position = 1`, `abstain = false`. The correct
  behaviour is "called **some** real function", i.e. `pred != NONE index`. Exact-match
  would under-report this, so the comparison runner scores relevance as **call-correct**
  (`pred != NONE index`), matching BFCL's relevance semantics.

**Why a NONE sentinel (vs. `gold_position = 0`):** the head's index decode and the STAR
contrastive geometry both need a real candidate span to point at and to push the abstain
representation away from. A sentinel candidate gives abstention a first-class span in the
exact same template; a magic `0` would need special-casing in the head's decode, the
batcher's span location, and the teacher. The sentinel keeps `prompt-template_sha8`
unchanged (same `render_messages`), so the trainer's drift guard still holds across all
four categories.

The sentinel's teacher cosine is `1.0` with itself (the irrelevance gold) and `~0` against
real function descriptions, so the STAR term pushes the abstain representation away from
the function cloud — the geometry we want.

## Commands

Prereqs (same as the rest of the repo): Apple Silicon + MLX; `BFCL_DATA` pointing at a
local BFCL checkout (the loader reads `$BFCL_DATA/data/BFCL_v4_<category>.json`); and, only
for **rebuilding** a split, `EMBEDDER_DIR` pointing at a local sentence embedder.

### (i) Build each category split

```bash
export BFCL_DATA=/path/to/BFCL
export EMBEDDER_DIR=/path/to/sentence-embedder   # only needed to rebuild a split

python data/build_bfcl_router_split.py --category live_multiple     # → data/bfcl_router/
python data/build_bfcl_router_split.py --category live_simple       # → data/bfcl_live_simple/
python data/build_bfcl_router_split.py --category live_relevance     # → data/bfcl_live_relevance/
python data/build_bfcl_router_split.py --category live_irrelevance   # → data/bfcl_live_irrelevance/
```

`live_multiple` rebuilds the committed `data/bfcl_router/` split byte-for-byte
(919 train / 134 holdout, `fn_desc_sha8=2c63c596`) — the entry-id hash and template are
unchanged. Each split writes `{train,holdout}.jsonl` + a `manifest.json` carrying
`category`, `gold_kind`, counts, and the `template_sha8` drift guard.

### (ii) Train a selector per category

Same trainer, just point `--data` at the per-category dir. Pick a `--save-dir` per
category so the adapters don't collide:

```bash
python -m train.trainer --head router --data data/bfcl_live_simple    \
  --base mlx-community/Qwen3-0.6B-bf16 --save-dir adapters/router-live_simple
python -m train.trainer --head router --data data/bfcl_live_relevance   \
  --base mlx-community/Qwen3-0.6B-bf16 --save-dir adapters/router-live_relevance
python -m train.trainer --head router --data data/bfcl_live_irrelevance \
  --base mlx-community/Qwen3-0.6B-bf16 --save-dir adapters/router-live_irrelevance
```

(`live_multiple` already has its committed adapter, `adapters/m1-router-star/`.)

> Note: the trainer's eval-gated early-stop uses **exact-match** holdout accuracy
> (`heads.router.eval`). For `live_relevance` that metric under-reports (it can't apply the
> call-correct rule), so the saved adapter is best-on-exact-match, not best-on-call-correct.
> The headline numbers should come from the **comparison runner** below, which applies the
> category-aware rule. For relevance specifically, consider it a directional result given
> the tiny n=16.

### (iii) Run the across-category comparison

Vanilla-only (no adapters — establishes the baselines across all four families):

```bash
python eval/bfcl/run_category_comparison.py \
  --base mlx-community/Qwen3-0.6B-bf16 --label 0.6B \
  --category live_multiple:data/bfcl_router \
  --category live_simple:data/bfcl_live_simple \
  --category live_relevance:data/bfcl_live_relevance \
  --category live_irrelevance:data/bfcl_live_irrelevance
```

Vanilla **vs** trained (append `:adapter` to each category that has one):

```bash
python eval/bfcl/run_category_comparison.py \
  --base mlx-community/Qwen3-0.6B-bf16 --label 0.6B \
  --category live_multiple:data/bfcl_router:adapters/m1-router-star/seed1/adapters.safetensors \
  --category live_simple:data/bfcl_live_simple:adapters/router-live_simple/seed0/adapters.safetensors \
  --category live_relevance:data/bfcl_live_relevance:adapters/router-live_relevance/seed0/adapters.safetensors \
  --category live_irrelevance:data/bfcl_live_irrelevance:adapters/router-live_irrelevance/seed0/adapters.safetensors
```

Each `--category` is `name:data_dir[:adapter]`; omit the adapter for a vanilla-only row.
The runner writes `eval/bfcl/category_comparison_<label>.json` and a matching
`category_comparison_<label>.md` (the table, paste-ready), and prints:

```
| category | n | vanilla acc | trained acc | Δ |
|---|---:|---:|---:|---:|
| live_multiple    | 134 | ... | ... | ... |
| live_simple      |  41 | ... | ... | ... |
| live_relevance   |   5 | ... | ... | ... |
| live_irrelevance | 127 | ... | ... | ... |
```

## Scoring rules (what the table means)

- `live_multiple`, `live_simple`: **exact-match** — `pred == gold_position`.
- `live_irrelevance`: **abstain-correct** — `pred == NONE index` (== `gold_position`).
- `live_relevance`: **call-correct** — `pred != NONE index` (chose some real function).

All four use the **same** greedy 1-based index decode (`heads.router._predict_index`) and
the **same** base load + DoRA install path (`shared.base`) as `router_eval.py`, so the
numbers are directly comparable to the committed `router_curve_*.json`.

## What this means for Paper 1 Limitation (c)

The single comparison table is the artifact that answers (c). Reading it:

- A positive **Δ on `live_simple`** (alongside `live_multiple`) shows the cheap-head win is
  not a `live_multiple` artifact — it holds on the easy K=1 family too.
- The **abstention columns** test a qualitatively different skill (when to call *nothing*).
  A trained-over-vanilla improvement on `live_irrelevance` (fewer hallucinated calls) and a
  non-regression on `live_relevance` (still calls when it should) shows the selector head
  generalizes beyond pure discrimination to the call/abstain decision.
- Caveat to report honestly: `live_relevance` holdout is **tiny** (n≈5 at 15% holdout, 16
  rows total), so treat its column as directional, not a headline number. `live_simple` (41)
  and `live_irrelevance` (127) are large enough to stand on their own.

Together these turn "shown on one task family" into "shown across four BFCL families with a
consistent harness", which is exactly what Limitation (c) asks for.
```
