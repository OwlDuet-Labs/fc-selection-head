# Data-scaling study — does the trained selector hit a *task* ceiling or a *data* ceiling?

This harness answers **Paper 1, Limitation (b)**.

## Why

The paper's §3 "training ceiling" claim — that the trained 0.6B selector plateaus and
more *adapter capacity* does not help — is currently defended by an **adapter-capacity
ablation** (varying DoRA rank / number of layers; see `router_capacity_*.json`). That
ablation rules out an *adapter* bottleneck. It does **not** rule out a *training-data*
bottleneck: the selector trains on only **919** BFCL `live_multiple` instances. If the
plateau is really a data ceiling, the capacity ablation would look flat regardless —
because the *data*, not the adapter, is the binding constraint.

The missing control is a **data-scaling curve**: train the same router head on
increasingly large slices of the corpus, eval each on the **same** holdout, and read off
the shape of trained-vs-vanilla accuracy.

- **Curve FLAT** (trained accuracy and the trained−vanilla delta stop moving well before
  919): the ceiling is a genuine **task/base ceiling**. Limitation (b) is answered — the
  result was *not* data-limited, and the §3 claim stands on a second, independent leg.
- **Curve still RISING at 919**: the §3 ceiling was a **training-data ceiling**. The
  adapter-capacity ablation alone did not settle it, and the paper should say so.

## What the harness produces

| File | Role |
|---|---|
| `data/build_datascaling_subsets.py` | writes `data/bfcl_router_n<size>/` subsets (train slice + shared holdout + manifest) |
| `eval/bfcl/run_datascaling.py` | `--plan` prints the command sequence; `--collect` assembles `router_datascaling.json` |
| `figures/fig_datascaling.py` | renders `figures/fig_datascaling_paper.png` from that JSON (placeholder if absent) |
| `eval/bfcl/router_datascaling.json` | the collected curve `{size: {vanilla_acc, trained_acc_mean, min, max, per_seed, delta}}` |

### Sampling (so small subsets are not degenerate)

Subsets are **deterministic**, **stratified**, and **nested**:

- **Deterministic** — each instance is keyed by `sha256(prompt + gold + K)`, not a Python
  RNG, so the subsets regenerate identically on any machine.
- **Stratified by gold function** — instances are ordered by a *coverage rank* assigned
  round-robin across the 193 distinct gold functions (round 0 takes the lowest-hash
  instance of every gold function, round 1 the next of each, …). The first 193 selected
  rows therefore cover every gold function exactly once; no gold function disappears until
  a subset is smaller than 193.
- **Nested** — a size-`M` subset is exactly the `M` lowest-coverage-rank rows, so
  `n100 ⊂ n250 ⊂ n500 ⊂ n919`. Each larger subset is a strict superset of the smaller, so
  the curve isolates "more data" with no resampling confound. (`n919` reproduces the full
  split, reordered — a no-op self-check.)

Every subset ships the **same** `holdout.jsonl` (the committed split's holdout, by default
symlinked; `--copy-holdout` to copy), so `router_eval.py` / the trainer's holdout metric
scores all sizes against one fixed test set.

## Exact command sequence

All commands run from the repo root (`fc-selection-head/`). Training and eval load MLX;
the subset builder, the collector, and the figure script do **not**.

```bash
# 1. Build the subsets (default sizes 100, 250, 500, 919).
python data/build_datascaling_subsets.py
#    or:  --sizes 100,250,500,919   |   --fractions 0.1,0.25,0.5,1.0   |   --copy-holdout

# 2. See the exact per-subset train + eval commands (does not run MLX):
python eval/bfcl/run_datascaling.py --plan
#    paste-and-run what it prints, or follow the pattern below.

# 2a. The size-independent vanilla baseline — run ONCE (vanilla base sees no train data):
python eval/bfcl/router_eval.py \
    --base mlx-community/Qwen3-0.6B-bf16 --label vanilla-0.6B

# 2b. Train an adapter per subset, 3 seeds each (HOST-STABILITY: one MLX process at a
#     time — do NOT parallelize; see train/trainer.py _install_memory_guards):
python -m train.trainer --head router --data data/bfcl_router_n100 \
    --base mlx-community/Qwen3-0.6B-bf16 --seeds 0,1,2 \
    --save-dir adapters/router-datascaling-n100
python -m train.trainer --head router --data data/bfcl_router_n250 \
    --base mlx-community/Qwen3-0.6B-bf16 --seeds 0,1,2 \
    --save-dir adapters/router-datascaling-n250
python -m train.trainer --head router --data data/bfcl_router_n500 \
    --base mlx-community/Qwen3-0.6B-bf16 --seeds 0,1,2 \
    --save-dir adapters/router-datascaling-n500
python -m train.trainer --head router --data data/bfcl_router_n919 \
    --base mlx-community/Qwen3-0.6B-bf16 --seeds 0,1,2 \
    --save-dir adapters/router-datascaling-n919

# 3. Collect into one curve JSON (no MLX). Reads each subset's trainer summary.json
#    (per-seed holdout selection accuracy) + the vanilla baseline JSON:
python eval/bfcl/run_datascaling.py --collect --vanilla-label vanilla-0.6B
#    -> eval/bfcl/router_datascaling.json

# 4. Render the figure (no MLX):
python figures/fig_datascaling.py
#    -> figures/fig_datascaling_paper.png
```

`--collect` degrades gracefully: any subset without a trained `summary.json` is reported
`UNTRAINED` and the JSON/figure show a partial curve, so you can checkpoint mid-run.

### Where the numbers come from

- **Trained accuracy** per size = the trainer's `summary.json → per_seed[*].holdout`
  (selection accuracy on the shared holdout), reported as a 3-seed mean with min/max.
  This is the same metric and harness as the existing curve JSONs — no second eval pass
  is required because the trainer already scores the full holdout when `eval_max_n ≥ 134`.
  (If you prefer an explicit, independent eval, run `router_eval.py --adapter
  adapters/router-datascaling-n<size>/seed<best>/adapters.safetensors` per size; the
  collector reads trainer summaries by default.)
- **Vanilla accuracy** = the single `router_curve_vanilla-0.6B.json` (`selection_acc`).
  The vanilla base never trains, so this one number is the flat reference at every size.

## Reading the result

Open `figures/fig_datascaling_paper.png` (trained mean + 3-seed band vs. the dashed
vanilla line, log-x) or inspect `router_datascaling.json`'s `delta` column:

| Shape | Conclusion for Paper 1 |
|---|---|
| `delta` flat from ~250 onward, band overlapping | **task/base ceiling** — Limitation (b) closed; not data-limited |
| `delta` still climbing at n=919 | **training-data ceiling** — capacity ablation insufficient; report the open scaling slope |

Either way the curve is a direct, reviewer-legible answer to "was the trained ceiling a
data ceiling?" that the capacity ablation cannot give on its own.
