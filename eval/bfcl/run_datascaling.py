#!/usr/bin/env python3
"""Orchestrate + collect the Paper 1 data-scaling study (Limitation b control).

Two modes, no MLX of its own — it only PRINTS the train/eval commands the dev runs,
then COLLECTS their JSON outputs into one curve file:

    --plan     enumerate the subset dirs and print the exact train + eval command
               sequence (one block per subset+seed). The dev runs these (they load
               MLX); this script never does.
    --collect  after the dev has trained an adapter per subset and produced the vanilla
               baseline eval, read every subset's trainer summary.json + the shared
               vanilla router_curve JSON and emit
               eval/bfcl/router_datascaling.json:
                   {size: {vanilla_acc, trained_acc_mean, trained_acc_min,
                           trained_acc_max, trained_per_seed, delta, n_seeds}}

The trained accuracy per subset is read from the trainer's `summary.json` (its
`per_seed[*].holdout` is selection accuracy on the SAME committed holdout that every
subset shares — that is exactly the comparable metric). The vanilla baseline is a
single number (vanilla base never sees train data, so it is size-independent) read
from a `router_curve_<vanilla-label>.json` produced by `eval/bfcl/router_eval.py`.

How to read the result (full rationale in eval/bfcl/DATASCALING_README.md):
  - delta FLAT across sizes  -> trained ceiling is a genuine task/base ceiling
                                (Limitation b answered: NOT data-limited).
  - delta RISES with size    -> the §3 ceiling was a TRAINING-DATA ceiling; the
                                adapter-capacity ablation alone did not settle it.

Usage:
    python eval/bfcl/run_datascaling.py --plan
    python eval/bfcl/run_datascaling.py --plan --seeds 0,1,2 --base mlx-community/Qwen3-0.6B-bf16
    python eval/bfcl/run_datascaling.py --collect
    python eval/bfcl/run_datascaling.py --collect --vanilla-label vanilla-0.6B
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "data"
ADAPTER_ROOT = REPO / "adapters"
EVAL_DIR = REPO / "eval" / "bfcl"

DEFAULT_BASE = "mlx-community/Qwen3-0.6B-bf16"
DEFAULT_SEEDS = (0, 1, 2)
OUT = EVAL_DIR / "router_datascaling.json"

_SUBSET_RE = re.compile(r"^bfcl_router_n(\d+)$")


def discover_subsets() -> list[tuple[int, Path]]:
    """Return [(size, dir)] for every data/bfcl_router_n<size>/ subset, sorted by size."""
    out = []
    for p in sorted(DATA_ROOT.glob("bfcl_router_n*")):
        m = _SUBSET_RE.match(p.name)
        if m and p.is_dir():
            out.append((int(m.group(1)), p))
    out.sort(key=lambda t: t[0])
    return out


def adapter_dir(size: int) -> Path:
    """Convention for a subset's trained-adapter save dir (matches --plan output)."""
    return ADAPTER_ROOT / f"router-datascaling-n{size}"


# ---------------------------------------------------------------------------- plan


def emit_plan(subsets, base: str, seeds, vanilla_label: str):
    """Print the exact, copy-pasteable command sequence the dev runs (no MLX here)."""
    if not subsets:
        print("No subsets found. Run `python data/build_datascaling_subsets.py` first.",
              file=sys.stderr)
        sys.exit(2)

    seed_str = ",".join(str(s) for s in seeds)
    print("# ===== data-scaling study — dev command sequence =====")
    print("# 0. (once) the size-independent vanilla baseline:")
    print(f"python eval/bfcl/router_eval.py --base {base} --label {vanilla_label}\n")

    for size, d in subsets:
        save = adapter_dir(size).relative_to(REPO)
        data_rel = d.relative_to(REPO)
        print(f"# ----- n={size} ({data_rel}) -----")
        print(f"python -m train.trainer --head router --data {data_rel} \\")
        print(f"    --base {base} --seeds {seed_str} --save-dir {save}")
        print(f"#   (trainer writes {save}/summary.json with per-seed holdout acc)\n")

    print("# ===== then collect =====")
    print(f"python eval/bfcl/run_datascaling.py --collect --vanilla-label {vanilla_label}")
    print("python figures/fig_datascaling.py   # render the curve")


# ------------------------------------------------------------------------- collect


def _vanilla_acc(vanilla_label: str):
    """Read the single size-independent vanilla selection-acc (%) if present."""
    p = EVAL_DIR / f"router_curve_{vanilla_label}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("selection_acc")


def _trained_accs(size: int):
    """Per-seed trained selection-acc (%) for a subset, from its trainer summary.json.

    Returns [] when the adapter has not been trained yet (so --collect degrades to a
    partial curve rather than failing).
    """
    summ = adapter_dir(size) / "summary.json"
    if not summ.exists():
        return []
    data = json.loads(summ.read_text())
    # trainer stores holdout as a fraction in [0,1]; the study reports percent.
    return [round(s["holdout"] * 100, 2) for s in data.get("per_seed", [])]


def collect(subsets, vanilla_label: str) -> dict:
    """Assemble {size: {...}} from trainer summaries + the vanilla baseline JSON."""
    vanilla = _vanilla_acc(vanilla_label)
    curve: dict[str, dict] = {}
    for size, _ in subsets:
        accs = _trained_accs(size)
        entry: dict = {"vanilla_acc": vanilla, "n_seeds": len(accs)}
        if accs:
            mean = round(sum(accs) / len(accs), 2)
            entry.update({
                "trained_acc_mean": mean,
                "trained_acc_min": min(accs),
                "trained_acc_max": max(accs),
                "trained_per_seed": accs,
                "delta": (round(mean - vanilla, 2) if vanilla is not None else None),
            })
        else:
            entry.update({
                "trained_acc_mean": None, "trained_acc_min": None,
                "trained_acc_max": None, "trained_per_seed": [],
                "delta": None, "status": "untrained",
            })
        curve[str(size)] = entry
    return {
        "study": "datascaling",
        "limitation": "Paper 1 (b) — training-DATA ceiling control",
        "vanilla_label": vanilla_label,
        "vanilla_acc": vanilla,
        "sizes": [s for s, _ in subsets],
        "per_size": curve,
    }


def emit_collect(subsets, vanilla_label: str):
    if not subsets:
        print("No subsets found. Run `python data/build_datascaling_subsets.py` first.",
              file=sys.stderr)
        sys.exit(2)
    result = collect(subsets, vanilla_label)
    OUT.write_text(json.dumps(result, indent=2))

    print(f"=== data-scaling collection -> {OUT.relative_to(REPO)} ===")
    van = result["vanilla_acc"]
    print(f"  vanilla ({vanilla_label}): "
          f"{van:.1f}%" if van is not None else f"  vanilla ({vanilla_label}): MISSING")
    for size in result["sizes"]:
        e = result["per_size"][str(size)]
        if e["trained_acc_mean"] is None:
            print(f"  n{size:<4} trained=UNTRAINED")
        else:
            d = e["delta"]
            print(f"  n{size:<4} trained={e['trained_acc_mean']:.1f}% "
                  f"[{e['trained_acc_min']:.1f}-{e['trained_acc_max']:.1f}] "
                  f"(seeds={e['n_seeds']}) "
                  f"delta={d:+.1f}" if d is not None else
                  f"  n{size:<4} trained={e['trained_acc_mean']:.1f}% (no vanilla)")
    missing = [s for s in result["sizes"]
               if result["per_size"][str(s)]["trained_acc_mean"] is None]
    if missing or van is None:
        print("\n  NOTE: partial curve — "
              + (f"vanilla baseline missing; " if van is None else "")
              + (f"untrained sizes {missing}; " if missing else "")
              + "rerun --plan, train the missing pieces, then --collect again.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true",
                      help="print the dev train+eval command sequence per subset")
    mode.add_argument("--collect", action="store_true",
                      help="collect trainer summaries + vanilla baseline into one curve JSON")
    ap.add_argument("--base", default=DEFAULT_BASE, help="base model id for the plan")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="comma list of seeds (default 0,1,2)")
    ap.add_argument("--vanilla-label", default="vanilla-0.6B",
                    help="label of the vanilla router_curve_<label>.json baseline")
    args = ap.parse_args()

    subsets = discover_subsets()
    if args.plan:
        seeds = tuple(int(s) for s in args.seeds.split(","))
        emit_plan(subsets, args.base, seeds, args.vanilla_label)
    else:
        emit_collect(subsets, args.vanilla_label)


if __name__ == "__main__":
    main()
