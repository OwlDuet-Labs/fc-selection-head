#!/usr/bin/env python3
"""Build deterministic, stratified, NESTED training subsets for the data-scaling study.

Paper 1 Limitation (b): the §3 "training ceiling" claim is defended by an
adapter-CAPACITY ablation (rank/layers), which rules out an adapter bottleneck but
NOT a training-DATA one — the selector trains on only 919 BFCL `live_multiple`
instances. This script produces the inputs for the missing data-scaling control:
train the same router head on increasingly large slices of the committed corpus,
eval each, and read off whether trained-vs-vanilla keeps RISING with more data (the
result was data-limited) or stays FLAT (a genuine task/base ceiling).

From the committed `data/bfcl_router/train.jsonl` (919 rows) it writes one dir per
target size:

    data/bfcl_router_n<size>/
        train.jsonl    # the subsampled training slice (size rows)
        holdout.jsonl  # the SAME holdout as the full split (copied, comparable eval)
        manifest.json  # provenance: parent split, size, sampling, gold coverage

The holdout is identical across every subset so `eval/bfcl/router_eval.py` scores all
sizes against one fixed test set — the only thing that varies is how much TRAIN data
the adapter saw.

Sampling (deterministic, stratified, nested)
--------------------------------------------
- **Deterministic**: each instance gets a stable key = sha256(prompt + gold + str(K)),
  not a Python RNG — the same subsets regenerate on any machine, any Python build.
- **Stratified by gold function**: instances are grouped by their gold function and
  assigned a *coverage rank* via round-robin across gold functions — round 0 takes the
  lowest-hash instance of EVERY gold function, round 1 the next, and so on. The first
  `n_gold_functions` (193) selected rows therefore cover every gold function exactly
  once, so even the smallest subset is not degenerate (no gold function vanishes until
  the subset is smaller than the number of gold functions).
- **Nested**: a subset of size M is exactly the M lowest-coverage-rank rows. Hence
  n100 ⊂ n250 ⊂ n500 ⊂ n919 — each larger subset is a strict superset of the smaller,
  so the scaling curve isolates "more data" with no resampling confound.

The full size (919) subset is written too; its train.jsonl is row-for-row the parent
(in coverage-rank order) — a self-check that the pipeline is a faithful no-op at 1.0.

Usage:
    python data/build_datascaling_subsets.py                       # default 100,250,500,919
    python data/build_datascaling_subsets.py --sizes 100,250,500,919
    python data/build_datascaling_subsets.py --fractions 0.1,0.25,0.5,1.0
    python data/build_datascaling_subsets.py --copy-holdout         # copy instead of symlink

Writes (per size) data/bfcl_router_n<size>/{train,holdout}.jsonl + manifest.json.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path

SRC = Path(__file__).resolve().parent / "bfcl_router"
OUT_ROOT = Path(__file__).resolve().parent


def _instance_key(inst: dict) -> str:
    """Stable per-instance hash key (hex sha256), independent of file/list order.

    Built from the fields that identify the instance content — prompt, gold function,
    and the full candidate-id list (in presentation order) — so the ordering is
    reproducible across machines and Python builds (unlike id() or a seeded RNG whose
    stream can drift between versions). The candidate list is included because a handful
    of corpus rows share a (prompt, gold, K) triple but differ in candidate set/order;
    folding the candidate ids in keeps the key 1:1 with the row's content.
    """
    cand_ids = "│".join(c["id"] for c in inst["candidates"])
    payload = (f"{inst['prompt']}␟{inst['gold']}␟"
               f"{inst.get('K', len(inst['candidates']))}␟{cand_ids}")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def coverage_ranked(train: list[dict]) -> list[dict]:
    """Order `train` by stratified coverage rank (round-robin across gold functions).

    Round 0 yields the lowest-hash instance of every distinct gold function, round 1
    the next-lowest of each, and so on. Truncating the returned list at N therefore
    keeps gold-function coverage maximal for that N and is strictly nested in N.
    """
    by_gold: dict[str, list[dict]] = collections.defaultdict(list)
    for inst in train:
        by_gold[inst["gold"]].append(inst)
    # Within each gold function, order by stable hash; across functions, order by name
    # so the round-robin pop order is itself deterministic.
    for insts in by_gold.values():
        insts.sort(key=_instance_key)
    golds = sorted(by_gold)

    ranked: list[dict] = []
    round_idx = 0
    remaining = sum(len(v) for v in by_gold.values())
    while remaining:
        for g in golds:
            bucket = by_gold[g]
            if round_idx < len(bucket):
                ranked.append(bucket[round_idx])
                remaining -= 1
        round_idx += 1
    return ranked


def _gold_coverage(rows: list[dict]) -> int:
    return len({r["gold"] for r in rows})


def write_subset(ranked: list[dict], size: int, holdout_path: Path,
                 parent_manifest: dict, copy_holdout: bool) -> Path:
    """Write data/bfcl_router_n<size>/ (train slice + shared holdout + manifest)."""
    out_dir = OUT_ROOT / f"bfcl_router_n{size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    subset = ranked[:size]

    (out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in subset) + "\n")

    # Holdout must be byte-identical to the parent split so every size is scored on the
    # same test set. Symlink by default (cheap, obvious provenance); copy on request
    # (portability / git-tracking of the subset dir).
    dest_holdout = out_dir / "holdout.jsonl"
    if dest_holdout.exists() or dest_holdout.is_symlink():
        dest_holdout.unlink()
    if copy_holdout:
        shutil.copyfile(holdout_path, dest_holdout)
    else:
        # Relative symlink so the tree stays relocatable.
        import os
        rel = os.path.relpath(holdout_path, out_dir)
        dest_holdout.symlink_to(rel)

    (out_dir / "manifest.json").write_text(json.dumps({
        "study": "datascaling",
        "limitation": "Paper 1 (b) — training-DATA ceiling control",
        "parent_split": str(SRC.relative_to(OUT_ROOT.parent)),
        "parent_n_train": parent_manifest.get("n_train"),
        "size": len(subset),
        "n_holdout": sum(1 for _ in open(holdout_path)),
        "holdout_mode": "copy" if copy_holdout else "symlink",
        "n_gold_functions_covered": _gold_coverage(subset),
        "n_gold_functions_total": _gold_coverage(ranked),
        "sampling": "deterministic sha256(prompt+gold+K) hash; "
                    "stratified round-robin over gold functions; "
                    "nested (size-M subset == M lowest coverage-rank rows)",
        "source": parent_manifest.get("source"),
        "fn_desc_sha8": parent_manifest.get("fn_desc_sha8"),
        "star_temp": parent_manifest.get("star_temp"),
    }, indent=2))
    return out_dir


def resolve_sizes(args, n_total: int) -> list[int]:
    """Turn --sizes / --fractions into a sorted, de-duplicated, clamped size list."""
    if args.fractions:
        fracs = [float(f) for f in args.fractions.split(",")]
        sizes = [max(1, round(f * n_total)) for f in fracs]
    else:
        sizes = [int(s) for s in args.sizes.split(",")]
    sizes = sorted({min(s, n_total) for s in sizes})
    return sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="100,250,500,919",
                    help="comma list of absolute training-set sizes (default 100,250,500,919)")
    ap.add_argument("--fractions", default="",
                    help="comma list of fractions of the full split, e.g. 0.1,0.25,0.5,1.0 "
                         "(overrides --sizes)")
    ap.add_argument("--copy-holdout", action="store_true",
                    help="copy holdout.jsonl into each subset dir instead of symlinking")
    args = ap.parse_args()

    train = [json.loads(l) for l in open(SRC / "train.jsonl")]
    holdout_path = SRC / "holdout.jsonl"
    parent_manifest = json.loads((SRC / "manifest.json").read_text())
    n_total = len(train)

    ranked = coverage_ranked(train)
    assert len(ranked) == n_total, "coverage ranking dropped/duplicated rows"

    sizes = resolve_sizes(args, n_total)
    print(f"parent: {n_total} train rows, {_gold_coverage(train)} gold functions, "
          f"{sum(1 for _ in open(holdout_path))} holdout rows")
    print(f"sizes:  {sizes}")

    for size in sizes:
        out_dir = write_subset(ranked, size, holdout_path, parent_manifest,
                               args.copy_holdout)
        subset = ranked[:size]
        print(f"  n{size:<4} -> {out_dir.relative_to(OUT_ROOT.parent)}  "
              f"({_gold_coverage(subset)}/{_gold_coverage(train)} gold fns covered)")

    print("\nNext: train an adapter per subset+seed, then run "
          "eval/bfcl/run_datascaling.py (see eval/bfcl/DATASCALING_README.md).")


if __name__ == "__main__":
    main()
