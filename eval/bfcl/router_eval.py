"""Standalone router selection-accuracy eval on data/bfcl_router/holdout.

Loads a base (any size) + optional trained DoRA adapter, runs the router head's
selection-accuracy eval. Used to build the trained-vs-vanilla scaling curve for Fig 1
with one consistent harness across 0.6B / 1.7B / 4B. Memory-guarded.

Usage:
  python eval/bfcl/router_eval.py --base mlx-community/Qwen3-1.7B-bf16 --label vanilla-1.7B
  python eval/bfcl/router_eval.py --base ... --adapter adapters/.../adapters.safetensors --label trained-1.7B
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
from mlx.utils import tree_unflatten

from shared import base
import heads.router as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="data/bfcl_router")
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-n", type=int, default=200)
    ap.add_argument("--label", default="model")
    args = ap.parse_args()

    mx.set_memory_limit(48 * 1024**3)   # bigger bases need more headroom than 24GB
    mx.set_cache_limit(2 * 1024**3)

    model, tok = base.load_base(args.base)
    if args.adapter:
        # Read the adapter's actual DoRA shape from its adapter_config.json (the trainer
        # saves rank/num_layers there). Falls back to CLI args. This is the fix for the
        # scaled-head eval bug: scaled adapters are rank 64 / 10 layers, not the 16/5
        # CLI default → installing the wrong shape made model.update() fail on param "m".
        rank, num_layers = args.rank, args.num_layers
        cfg_path = Path(args.adapter).parent / "adapter_config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            rank = int(cfg.get("rank", rank))
            num_layers = int(cfg.get("num_layers", num_layers))
        base.install_dora(model, num_layers, rank)
        weights = mx.load(args.adapter)
        model.update(tree_unflatten(list(weights.items())))
        mx.eval(model.parameters())
    model.eval()

    holdout = [json.loads(l) for l in open(Path(args.data) / "holdout.jsonl")]
    acc = R.eval(model, tok, holdout, max_n=args.max_n)
    res = {"label": args.label, "base": args.base, "adapter": args.adapter,
           "selection_acc": acc * 100, "n": min(len(holdout), args.max_n)}
    print(f"=== router eval — {args.label}: selection-acc {acc*100:.1f}% (n={res['n']}) ===")
    out = Path("eval/bfcl") / f"router_curve_{args.label}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
