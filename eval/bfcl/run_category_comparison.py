#!/usr/bin/env python3
"""Across-BFCL-categories selector comparison — Paper 1 Limitation (c) closer.

Runs the selection-head eval over MULTIPLE BFCL categories with ONE consistent
harness and emits a single comparison JSON + a printed markdown table
(category | n | vanilla acc | trained acc | delta). This generalizes the
single-category `router_eval.py` curve to the across-task-family table the paper
needs: the headline claim ("a trained 0.6B selector matches a vanilla model 7x
its size") is currently shown only on `live_multiple`; this runner shows whether
the trained-over-vanilla delta holds on live_simple (easy, K=1) and on the
abstention categories (live_relevance / live_irrelevance).

Per category, for the same base, it scores:
  * VANILLA  — base, no adapter.
  * TRAINED  — base + the per-category trained DoRA adapter (if supplied).
The eval reuses heads.router (the selection head) and shared.base (load + DoRA
install) IN-PROCESS — same code path as router_eval.py — so the numbers are
directly comparable to the committed router_curve_*.json.

ABSTENTION SCORING (load-bearing — see data/build_bfcl_router_split.py):
  The builder appends a sentinel "NONE" candidate to relevance/irrelevance
  instances. Categories are scored with DIFFERENT correctness rules:
    * named  (live_multiple/live_simple): exact-match — pred == gold_position.
    * irrelevance: ABSTAIN-correct — pred == the NONE index (gold_position).
                   (Exact-match already gives this, since gold IS NONE.)
    * relevance:   CALL-correct — pred != the NONE index (model called SOME real
                   function). There is no single named gold index for relevance,
                   so exact-match would UNDER-report it; this rule matches BFCL's
                   relevance semantics ("output a function call, any valid one").
  The per-instance NONE index is `inst['K']` (NONE is appended last) for
  abstention categories. heads.router.eval cannot express the relevance rule, so
  this runner reimplements the thin scoring loop around heads.router's
  `_predict_index` (the SAME greedy index decode) rather than calling its `eval`.

Heavy imports (mlx / mlx_lm via shared.base + heads.router) are deferred into
`run()` so `--help` and ast/import smoke tests never load a model.

Usage:
  # vanilla-only across all four categories (no adapters):
  python eval/bfcl/run_category_comparison.py \
      --base mlx-community/Qwen3-0.6B-bf16 \
      --category live_multiple:data/bfcl_router \
      --category live_simple:data/bfcl_live_simple \
      --category live_relevance:data/bfcl_live_relevance \
      --category live_irrelevance:data/bfcl_live_irrelevance \
      --label 0.6B

  # with per-category trained adapters (vanilla vs trained delta):
  python eval/bfcl/run_category_comparison.py \
      --base mlx-community/Qwen3-0.6B-bf16 \
      --category live_multiple:data/bfcl_router:adapters/m1-router-star/seed1/adapters.safetensors \
      --category live_simple:data/bfcl_live_simple:adapters/router-live_simple/seed0/adapters.safetensors \
      --label 0.6B
Writes eval/bfcl/category_comparison_<label>.json.
"""
import argparse
import json
import sys
from pathlib import Path

# Repo root on sys.path (mirror router_eval.py) — NO heavy import at module load.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def parse_category_spec(spec: str) -> dict:
    """`name:data_dir[:adapter]` → {name, data, adapter}.

    adapter is optional; an empty trailing field (`name:data:`) means vanilla-only.
    """
    parts = spec.split(":")
    if len(parts) < 2 or not parts[0] or not parts[1]:
        raise argparse.ArgumentTypeError(
            f"--category must be 'name:data_dir[:adapter]', got {spec!r}")
    name, data = parts[0], parts[1]
    adapter = parts[2] if len(parts) >= 3 and parts[2] else ""
    return {"name": name, "data": data, "adapter": adapter}


# Category → correctness rule. `none_index` rules consume the appended sentinel.
ABSTAIN_CATEGORIES = {"live_relevance", "live_irrelevance"}


def _score_category(model, tok, holdout, category, max_n):
    """(accuracy, n) under the category-aware correctness rule.

    Uses heads.router._predict_index (the same greedy 1-based index decode the
    standard eval uses) so the decode is identical across categories.
    """
    import heads.router as R
    import mlx.core as mx

    n = min(len(holdout), max_n)
    correct = 0
    for inst in holdout[:n]:
        pred = R._predict_index(model, tok, inst)
        none_index = inst["K"]  # NONE was appended last (abstention categories)
        if category == "live_relevance":
            # CALL-correct: chose SOME real function (not the abstain sentinel)
            # and produced a parseable index in range.
            ok = (pred != none_index) and (1 <= pred <= inst["K"])
        else:
            # named + irrelevance: exact-match against gold_position
            # (for irrelevance, gold_position == the NONE index).
            ok = (pred == int(inst["gold_position"]))
        correct += int(ok)
        mx.clear_cache()
    return correct / max(n, 1), n


def _eval_one(base, adapter, data_dir, category, num_layers, rank, max_n):
    """Load base (+optional adapter), score the category holdout. Heavy import
    path — only called inside run()."""
    import mlx.core as mx
    from mlx.utils import tree_unflatten
    from shared import base as B

    model, tok = B.load_base(base)
    if adapter:
        # Read the adapter's real DoRA shape from adapter_config.json (the trainer
        # writes rank/num_layers there); fall back to CLI args. Mirrors the
        # router_eval.py scaled-adapter fix.
        r, nl = rank, num_layers
        cfg_path = Path(adapter).parent / "adapter_config.json"
        if cfg_path.exists():
            cfg = json.load(open(cfg_path))
            r = int(cfg.get("rank", r))
            nl = int(cfg.get("num_layers", nl))
        B.install_dora(model, nl, r)
        weights = mx.load(adapter)
        model.update(tree_unflatten(list(weights.items())))
        mx.eval(model.parameters())
    model.eval()

    holdout = [json.loads(l) for l in open(Path(data_dir) / "holdout.jsonl")]
    return _score_category(model, tok, holdout, category, max_n)


def run(args) -> dict:
    """Evaluate every category (vanilla, and trained if an adapter is given) and
    return the comparison record. Heavy imports happen here, not at module load."""
    mx = __import__("mlx.core", fromlist=["core"])
    mx.set_memory_limit(48 * 1024**3)   # match router_eval.py headroom
    mx.set_cache_limit(2 * 1024**3)

    rows = []
    for spec in args.category:
        cat, data, adapter = spec["name"], spec["data"], spec["adapter"]
        print(f"\n=== {cat} (data={data}) ===", flush=True)

        van_acc, n = _eval_one(args.base, "", data, cat,
                               args.num_layers, args.rank, args.max_n)
        print(f"  vanilla: {van_acc*100:.1f}% (n={n})", flush=True)

        tr_acc = None
        if adapter:
            tr_acc, _ = _eval_one(args.base, adapter, data, cat,
                                  args.num_layers, args.rank, args.max_n)
            print(f"  trained: {tr_acc*100:.1f}% (n={n}) "
                  f"[Δ {(tr_acc-van_acc)*100:+.1f}]", flush=True)

        rows.append({
            "category": cat, "n": n, "data": data, "adapter": adapter,
            "vanilla_acc": round(van_acc * 100, 1),
            "trained_acc": (round(tr_acc * 100, 1) if tr_acc is not None else None),
            "delta": (round((tr_acc - van_acc) * 100, 1) if tr_acc is not None else None),
        })

    return {"label": args.label, "base": args.base, "categories": rows}


def render_markdown(record: dict) -> str:
    """Markdown comparison table (category | n | vanilla | trained | delta)."""
    lines = [
        f"### BFCL across-category selector comparison — {record['label']} "
        f"(base `{record['base']}`)",
        "",
        "| category | n | vanilla acc | trained acc | Δ |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in record["categories"]:
        tr = "—" if r["trained_acc"] is None else f"{r['trained_acc']:.1f}"
        dl = "—" if r["delta"] is None else f"{r['delta']:+.1f}"
        lines.append(
            f"| {r['category']} | {r['n']} | {r['vanilla_acc']:.1f} | {tr} | {dl} |")
    lines.append("")
    lines.append(
        "_Scoring: live_multiple/live_simple = exact-match (pred == gold index); "
        "live_irrelevance = abstain-correct (pred == NONE index); "
        "live_relevance = call-correct (pred != NONE index). "
        "See data/build_bfcl_router_split.py for the abstention convention._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Across-BFCL-categories selector comparison (Paper 1 Lim. c).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Each --category is 'name:data_dir[:adapter]'. Repeat per category. "
               "Omit the adapter (or leave it empty) for a vanilla-only row.")
    ap.add_argument("--base", required=True, help="base model id (mlx-community/...)")
    ap.add_argument("--category", action="append", type=parse_category_spec,
                    required=True, metavar="NAME:DATA[:ADAPTER]",
                    help="category spec; repeatable (one per BFCL category).")
    ap.add_argument("--num-layers", type=int, default=5, help="DoRA fallback shape")
    ap.add_argument("--rank", type=int, default=16, help="DoRA fallback rank")
    ap.add_argument("--max-n", type=int, default=200, help="cap holdout rows scored")
    ap.add_argument("--label", default="model", help="output label / table title")
    args = ap.parse_args()

    record = run(args)

    md = render_markdown(record)
    print("\n" + md + "\n")

    out = Path("eval/bfcl") / f"category_comparison_{args.label}.json"
    out.write_text(json.dumps(record, indent=2))
    print(f"  wrote {out}")
    # The markdown table lands beside the JSON for direct paste into the paper.
    out_md = out.with_suffix(".md")
    out_md.write_text(md + "\n")
    print(f"  wrote {out_md}")


if __name__ == "__main__":
    main()
