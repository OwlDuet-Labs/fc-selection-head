#!/usr/bin/env python3
"""Render the Paper 1 data-scaling figure (Limitation b control).

Plots trained-vs-vanilla BFCL selection accuracy against training-set size on a log-x
axis, reading eval/bfcl/router_datascaling.json (produced by
eval/bfcl/run_datascaling.py --collect). The trained curve carries a 3-seed min/max
band; the vanilla baseline is a flat reference line (the vanilla base sees no train
data, so it is size-independent).

Reading the figure:
  - trained curve FLAT (band overlaps the same level across sizes) -> the trained
    ceiling is a genuine task/base ceiling; Limitation (b) is answered (not data-limited).
  - trained curve still RISING at n=919 -> the §3 ceiling was a TRAINING-DATA ceiling
    and the adapter-capacity ablation alone did not settle it.

No MLX. Pure matplotlib over the collected JSON. If the JSON is absent or every size is
still untrained, a labelled PLACEHOLDER figure is rendered so the figure slot exists in
the build before the run finishes.

Usage:
    python figures/fig_datascaling.py
    python figures/fig_datascaling.py --json eval/bfcl/router_datascaling.json \
        --out figures/fig_datascaling_paper.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_JSON = REPO / "eval" / "bfcl" / "router_datascaling.json"
DEFAULT_OUT = REPO / "figures" / "fig_datascaling_paper.png"


def _placeholder(ax, msg: str):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
            fontsize=11, color="#888888", wrap=True)
    ax.set_xscale("log")
    ax.set_xlabel("training-set size (instances, log scale)")
    ax.set_ylabel("BFCL selection accuracy (%)")


def _trained_points(data: dict):
    """Return (sizes, means, mins, maxs) over only the sizes that have a trained mean."""
    sizes, means, lo, hi = [], [], [], []
    for s in data["sizes"]:
        e = data["per_size"][str(s)]
        if e.get("trained_acc_mean") is not None:
            sizes.append(s)
            means.append(e["trained_acc_mean"])
            lo.append(e["trained_acc_min"])
            hi.append(e["trained_acc_max"])
    return sizes, means, lo, hi


def render(json_path: Path, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")  # headless; no display needed for a saved PNG
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.6))

    if not json_path.exists():
        _placeholder(ax, f"no data yet\n(run eval/bfcl/run_datascaling.py --collect\n"
                         f"to write {json_path.name})")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"  wrote PLACEHOLDER {out_path} (no {json_path})")
        return

    data = json.loads(json_path.read_text())
    sizes, means, lo, hi = _trained_points(data)
    vanilla = data.get("vanilla_acc")

    if not sizes:
        _placeholder(ax, "subsets enumerated but no adapters trained yet\n"
                         "(train per subset, then --collect)")
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        print(f"  wrote PLACEHOLDER {out_path} (no trained sizes in {json_path.name})")
        return

    # trained curve + 3-seed min/max band
    ax.plot(sizes, means, marker="o", color="#1f6feb", label="trained (3-seed mean)")
    ax.fill_between(sizes, lo, hi, color="#1f6feb", alpha=0.15, label="3-seed min/max")

    # vanilla flat reference (size-independent)
    if vanilla is not None:
        ax.axhline(vanilla, color="#888888", linestyle="--",
                   label=f"vanilla ({data.get('vanilla_label', 'vanilla')}) {vanilla:.1f}%")

    ax.set_xscale("log")
    ax.set_xlabel("training-set size (instances, log scale)")
    ax.set_ylabel("BFCL selection accuracy (%)")
    ax.set_title("Function-call selector: data scaling")
    ax.set_xticks(sizes)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, which="both", alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"  wrote {out_path}  (sizes={sizes}, vanilla={vanilla})")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", type=Path, default=DEFAULT_JSON,
                    help="collected curve JSON (default eval/bfcl/router_datascaling.json)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help="output PNG (default figures/fig_datascaling_paper.png)")
    args = ap.parse_args()
    render(args.json, args.out)


if __name__ == "__main__":
    main()
