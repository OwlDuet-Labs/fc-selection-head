"""M1 acceptance gate: reproduce the BFCL STAR result from inside sqrl/.

Trains the router head two ways on data/bfcl_router with the self-contained
trainer — STAR (lambda=0.10, star_temp=0.08) and vanilla CE (lambda=0) — over
the same seeds, and reports holdout selection accuracy for each. The methods-
paper claim this validates: STAR lifts a small router over vanilla CE on BFCL
live_multiple (the result the paper reports), self-contained.

This is the port's proof-of-life. If STAR > vanilla here, the trainer/batch/head
port is faithful; the number itself feeds the methods paper's reproduction line.

Run as the ONLY MLX process (host-stability rule). ~minutes per config/seed.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import importlib
from dataclasses import replace

from train.config import TrainConfig
from train import trainer


def run():
    head = importlib.import_module("heads.router")
    data = "data/bfcl_router"
    seeds = (0, 1, 2)
    base_cfg = TrainConfig(seeds=seeds, max_steps=600, eval_every=50,
                           eval_max_n=134)  # full holdout

    configs = {
        "star_L010_T008": replace(base_cfg, star_lambda=0.10, star_temp=0.08,
                                  save_dir="adapters/m1-router-star"),
        "vanilla_ce":     replace(base_cfg, star_lambda=0.0,
                                  save_dir="adapters/m1-router-vanilla"),
    }

    results = {}
    for name, cfg in configs.items():
        print(f"\n########## {name} ##########", flush=True)
        best, path = trainer.train(head, data, cfg)
        results[name] = {"best_holdout": best, "adapter": str(path)}

    print("\n==================== M1 RESULT ====================")
    star = results["star_L010_T008"]["best_holdout"]
    van = results["vanilla_ce"]["best_holdout"]
    print(f"  STAR (L0.10/T0.08) holdout = {star:.3f}")
    print(f"  vanilla CE         holdout = {van:.3f}")
    print(f"  Δ (STAR - vanilla)         = {star - van:+.3f}")
    print(f"  {'STAR WINS — port reproduces the BFCL STAR effect' if star > van else 'NO STAR WIN — investigate'}")

    out = Path("eval/bfcl/m1_result.json")
    import json
    out.write_text(json.dumps({
        "star_holdout": star, "vanilla_holdout": van, "delta": star - van,
        "seeds": list(seeds), "data": data, "detail": results,
    }, indent=2))
    print(f"  wrote {out}")


if __name__ == "__main__":
    run()
