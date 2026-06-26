"""Paper Figure 3 — latency & memory on real hardware (self-contained, no external style dep).

The parameter-efficiency result (a trained 0.6B + head matches a vanilla 4B on accuracy) carries
to wall-clock and memory. Real Apple M1 (16 GB) sweep, batch-1 single-stream, in-process MLX:
Qwen3 0.6B/1.7B/4B at bf16/q6/q4.

  LEFT  — median ms/query (log y), grouped by base size, one bar per quant; p90 cap.
  RIGHT — peak memory (GB), same grouping.

Headline: trained 0.6B-q4 = 440 ms / 0.49 GB vs vanilla 4B-q4 = 2176 ms / 2.52 GB
(~4.9x faster, ~5.1x smaller at matched accuracy).

Reads eval/bfcl/latency_result.json. Usage: python figures/fig_latency.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RESULT = Path(__file__).resolve().parents[1] / "eval" / "bfcl" / "latency_result.json"
SIZES = ["0.6B", "1.7B", "4B"]
QUANTS = ["bf16", "q6", "q4"]
QCOL = {"bf16": "#6c7a89", "q6": "#e8702a", "q4": "#2e8b57"}  # neutral / accent / good


def main():
    if not RESULT.exists():
        print(f"fig_latency: no {RESULT} — copy the M1 sweep result first")
        return
    data = json.loads(RESULT.read_text())
    runs = {(r["label"], r["quant"]): r for r in data["all_runs"]}
    device = data.get("device", "device")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(SIZES))
    w = 0.26

    # LEFT: median latency (log) with p90 cap
    for qi, q in enumerate(QUANTS):
        med = [runs[(s, q)]["median_ms"] for s in SIZES]
        p90 = [runs[(s, q)]["p90_ms"] for s in SIZES]
        xs = x + (qi - 1) * w
        axL.bar(xs, med, width=w, color=QCOL[q], label=q)
        axL.errorbar(xs, med, yerr=[[0] * len(med), [p - m for p, m in zip(p90, med)]],
                     fmt="none", ecolor="#333", elinewidth=0.8, capsize=2, alpha=0.6)
        for xx, m in zip(xs, med):
            axL.annotate(f"{m}", (xx, m * 1.04), ha="center", va="bottom", fontsize=7, rotation=90)
    axL.set_yscale("log")
    axL.set_xticks(x); axL.set_xticklabels([f"{s}\nbase" for s in SIZES])
    axL.set_ylabel("median ms / query  (log, batch=1)")
    axL.set_title(f"A . Wall-clock latency on {device}\n"
                  "trained 0.6B (ship size) ~5x faster than vanilla 4B at matched accuracy")
    axL.legend(title="quant", frameon=False, fontsize=9)
    axL.grid(True, axis="y", alpha=0.25)

    # RIGHT: peak memory
    for qi, q in enumerate(QUANTS):
        mem = [runs[(s, q)]["peak_gb"] for s in SIZES]
        xs = x + (qi - 1) * w
        axR.bar(xs, mem, width=w, color=QCOL[q], label=q)
        for xx, mm in zip(xs, mem):
            axR.annotate(f"{mm:.2f}", (xx, mm + 0.08), ha="center", fontsize=7)
    axR.set_xticks(x); axR.set_xticklabels([f"{s}\nbase" for s in SIZES])
    axR.set_ylabel("peak memory (GB)")
    axR.set_title("B . Peak memory\ntrained 0.6B-q4 = 0.49 GB vs vanilla 4B-q4 = 2.52 GB")
    axR.legend(title="quant", frameon=False, fontsize=9)
    axR.grid(True, axis="y", alpha=0.25)

    fig.suptitle("Efficiency beyond parameters: the trained 0.6B head is ~5x faster and ~5x smaller "
                 "than the vanilla 4B it matches on accuracy (Apple M1, 16 GB, in-process)",
                 fontsize=11, weight="bold")
    out = Path(__file__).resolve().parent / "fig_latency_paper.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    sp = runs[("4B", "q4")]["median_ms"] / runs[("0.6B", "q4")]["median_ms"]
    mr = runs[("4B", "q4")]["peak_gb"] / runs[("0.6B", "q4")]["peak_gb"]
    print(f"wrote {out}  (0.6B-q4 vs 4B-q4: {sp:.1f}x faster, {mr:.1f}x smaller)")


if __name__ == "__main__":
    main()
