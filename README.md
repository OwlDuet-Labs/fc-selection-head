# fc-selection-head

Reproduction code for *Cheap Heads, Not Bigger Bases: A Trained 0.6B Function-Call
Selector Matches a Vanilla Model Seven Times Its Size* (under double-blind review; author
information withheld during review).

A frozen small base model plus a cheap trained **selector head** (a DoRA adapter, ~0.14%
of base parameters) matches a vanilla base seven times its size on BFCL `live_multiple`
function-call selection. This repository contains the self-contained training and
evaluation pipeline, the committed BFCL split, and the trained adapters behind the paper's
BFCL results.

## What's here

```
shared/          base-model loading + DoRA install, the contrastive losses, teacher targets
train/           the self-contained trainer (config, batching, loop)
heads/router/    the selector head (reads a candidate list, emits the selected index)
data/bfcl_router/  the committed BFCL live_multiple split (train.jsonl, holdout.jsonl)
data/build_bfcl_router_split.py   rebuilds the split from raw BFCL (optional)
eval/bfcl/       selection-accuracy eval + the trained/vanilla/capacity result JSONs
adapters/        trained selector heads (0.6B/1.7B/4B) + capacity-scaled (4B/14B) heads
```

## Requirements

- Apple Silicon (the pipeline uses Apple MLX); `pip install mlx mlx-lm numpy`
- The base weights are pulled from the MLX community on first run
  (`mlx-community/Qwen3-0.6B-bf16`, etc.).

## Reproduce the headline result

Evaluate a trained selector head (frozen 0.6B base + the committed adapter) on the BFCL
holdout:

```bash
python eval/bfcl/router_eval.py \
  --base mlx-community/Qwen3-0.6B-bf16 \
  --adapter adapters/m1-router-star/seed1/adapters.safetensors \
  --label trained-0.6B
```

Expected: ~91% selection accuracy (best seed; the paper reports a 3-seed mean of 90.3).
The vanilla baseline:

```bash
python eval/bfcl/router_eval.py --base mlx-community/Qwen3-0.6B-bf16 --label vanilla-0.6B
# ~80.6%
```

The full trained-vs-vanilla scaling curve and the capacity ablation are reproduced by
running the eval across base sizes / adapters; per-run results are committed under
`eval/bfcl/router_curve_*.json` and `eval/bfcl/router_capacity_*.json`.

## Train a selector head from scratch

```bash
python -m train.trainer --head router --data data/bfcl_router --base mlx-community/Qwen3-0.6B-bf16
```

Single-GPU MLX, roughly four minutes per seed. See `train/config.py` for the (small)
configuration surface.

## Notes

- This repository covers the BFCL **selector head** only. The REAPER transfer results
  (§5 of the paper) are eval-only under GPL-3.0 and are released separately with the
  accompanying dataset paper; they are not redistributed here.
- The committed BFCL split already includes the precomputed graded-teacher targets, so the
  data builder and a sentence-embedder are needed only to rebuild the split from raw BFCL.

## License

Apache-2.0 (see `LICENSE`).

## Citation

```bibtex
@misc{cheapheads2026,
  title  = {Cheap Heads, Not Bigger Bases: A Trained 0.6B Function-Call Selector
            Matches a Vanilla Model Seven Times Its Size},
  author = {Anonymous},
  year   = {2026},
  note   = {Under double-blind review}
}
```
