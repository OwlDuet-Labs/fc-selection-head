"""Position-confounder check for the selector head.

A reviewer will ask: is the 91% just a positional artifact (gold clusters at index 1 and the
head always picks 1)? This script refutes that on three axes:
  1. the GOLD index distribution in the holdout (and the 'always pick 1' / random baselines),
  2. the head's PICK distribution (does it just point at index 1?),
  3. accuracy BY gold position (is it only right when gold is early?).

Writes eval/bfcl/position_confounder.json.
"""
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
from mlx.utils import tree_unflatten

from shared import base
import heads.router as R


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="mlx-community/Qwen3-0.6B-bf16")
    ap.add_argument("--adapter", default="adapters/m1-router-star/seed1/adapters.safetensors")
    ap.add_argument("--data", default="data/bfcl_router")
    args = ap.parse_args()

    holdout = [json.loads(l) for l in open(Path(args.data) / "holdout.jsonl")]
    n = len(holdout)

    gold = collections.Counter(int(x["gold_position"]) for x in holdout)
    always1 = 100 * gold[1] / n
    random_exp = 100 * sum(1.0 / len(x["candidates"]) for x in holdout) / n

    model, tok = base.load_base(args.base)
    base.install_dora(model, 5, 16)
    model.update(tree_unflatten(list(mx.load(args.adapter).items())))
    mx.eval(model.parameters()); model.eval()

    picks = collections.Counter()
    by_goldpos = collections.defaultdict(lambda: [0, 0])
    correct = 0
    for inst in holdout:
        p = R._predict_index(model, tok, inst)
        g = int(inst["gold_position"])
        picks[p] += 1
        by_goldpos[g][1] += 1
        if p == g:
            by_goldpos[g][0] += 1
            correct += 1
        mx.clear_cache()

    acc = 100 * correct / n
    res = {
        "n": n, "selection_acc": acc,
        "baseline_always_pick_1": always1, "baseline_random": random_exp,
        "gold_position_dist": {str(k): gold[k] for k in sorted(gold)},
        "head_pick_dist": {str(k): picks[k] for k in sorted(picks)},
        "accuracy_by_gold_position": {str(g): {"correct": by_goldpos[g][0], "n": by_goldpos[g][1],
                                               "acc": 100 * by_goldpos[g][0] / by_goldpos[g][1]}
                                      for g in sorted(by_goldpos)},
    }
    print(f"acc={acc:.1f}%  always-pick-1={always1:.1f}%  random={random_exp:.1f}%")
    print("acc by gold pos:", {g: f"{v['acc']:.0f}%" for g, v in res["accuracy_by_gold_position"].items()})
    out = Path("eval/bfcl") / "position_confounder.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
