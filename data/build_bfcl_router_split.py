#!/usr/bin/env python3
"""Build a STAR router train/holdout split from BFCL live_multiple.

Reshapes BFCL `live_multiple` (1053 single-call entries, gold always among the
N candidate functions) into the selection-head trainer's instance format, with
STAR teacher cosines on each candidate. This is the publishable SQRL experiment's
data: train a STAR DoRA router on BFCL's OWN tasks and beat vanilla 0.6B on
BFCL's scorer.

Builds the BFCL live_multiple selector split (train/holdout JSONL).
  {prompt, candidates:[{id, desc, star_sim}], gold, gold_position(1-based), K, source}
  - id        = the BFCL function name (what the router emits/picks)
  - desc      = the function description (router candidate context + teacher text)
  - star_sim  = cos(emb(cand.desc), emb(gold.desc)) — the STAR graded teacher,
                built from the SAME frozen embedder, via sqrl/shared/teacher_targets.

Split: deterministic by entry id hash → train ~85% / holdout ~15%. Gold-function
recurrence (max 84) means the router can learn discrimination; the holdout has
held-out ENTRIES (same function vocabulary), the standard in-distribution test.

Teacher uses the function DESCRIPTIONS (not names): two functions are "near" if
their descriptions are semantically close — exactly the discrimination signal.

Usage: python build_bfcl_router_split.py [--embedder-dir DIR] [--holdout-frac 0.15]
Writes data/bfcl_router/{train,holdout}.jsonl + manifest.json.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import teacher_targets as TT

# Point BFCL_DATA at your local BFCL `live_multiple` source (the public gorilla/BFCL repo).
import os
BFCL = Path(os.environ.get("BFCL_DATA", "/path/to/BFCL/live_multiple"))
OUT = Path(__file__).resolve().parent / "bfcl_router"
# A local sentence-embedder for the STAR graded teacher (any HF/MLX sentence embedder);
# set EMBEDDER_DIR. The committed split already includes precomputed teacher targets, so
# this is only needed to rebuild the split from raw BFCL.
DEFAULT_EMBEDDER = Path(os.environ.get("EMBEDDER_DIR", "/path/to/sentence-embedder"))


def make_embed_fn(embedder_dir):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(embedder_dir), device="mps")

    def embed(texts):
        import numpy as np
        v = model.encode(list(texts), batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
        return v.astype("float32")
    return embed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder-dir", type=Path, default=DEFAULT_EMBEDDER)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--star-temp", type=float, default=0.08)
    args = ap.parse_args()

    d = [json.loads(l) for l in open(BFCL / "data/BFCL_v4_live_multiple.json")]
    gt = {g["id"]: g for g in
          (json.loads(l) for l in
           open(BFCL / "data/possible_answer/BFCL_v4_live_multiple.json"))}

    # 1. collect every (fn name -> description) across the corpus for the teacher.
    fn_desc = {}
    for e in d:
        for f in e["function"]:
            fn_desc.setdefault(f["name"], f.get("description", "") or f["name"])
    print(f"  {len(d)} entries, {len(fn_desc)} distinct functions", file=sys.stderr)

    # 2. embed every unique function description ONCE → teacher vectors.
    embed = make_embed_fn(args.embedder_dir)
    key2vec, sha = TT.build_text_vectors(fn_desc, embed)
    print(f"  embedded {len(key2vec)} fn descriptions (sha8={sha})", file=sys.stderr)

    # 3. build instances with STAR teacher cosines per candidate.
    train, hold = [], []
    n_skip = 0
    for e in d:
        if e["id"] not in gt:
            n_skip += 1
            continue
        gold = list(gt[e["id"]]["ground_truth"][0].keys())[0]
        cand_names = [f["name"] for f in e["function"]]
        if gold not in cand_names:
            n_skip += 1
            continue
        gold_pos = cand_names.index(gold) + 1  # 1-based
        sims = TT.teacher_for_instance(cand_names, gold, key2vec)  # cos to gold
        cands = [{"id": n, "desc": fn_desc[n][:120],
                  "star_sim": (sims[i] if sims else (1.0 if n == gold else 0.0))}
                 for i, n in enumerate(cand_names)]
        inst = {"prompt": e["question"][0][0]["content"], "candidates": cands,
                "gold": gold, "gold_position": gold_pos, "K": len(cands),
                "source": "bfcl_live_multiple"}
        # deterministic split by entry id
        h = int(hashlib.sha256(str(e["id"]).encode()).hexdigest(), 16) % 1000
        (hold if h < args.holdout_frac * 1000 else train).append(inst)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train) + "\n")
    (OUT / "holdout.jsonl").write_text(
        "\n".join(json.dumps(x) for x in hold) + "\n")
    (OUT / "manifest.json").write_text(json.dumps({
        "source": "BFCL_v4_live_multiple", "n_train": len(train),
        "n_holdout": len(hold), "n_skip": n_skip, "n_functions": len(fn_desc),
        "fn_desc_sha8": sha, "star_temp": args.star_temp,
        "holdout_frac": args.holdout_frac,
        # the selection-head trainer asserts manifest['template_sha8'] ==
        # train_selection_head.template_sha8() (prompt-template drift guard).
        # This is the current trainer value; re-derive if the template changes.
        "template_sha8": "58cc417d"}, indent=2))
    print(f"\nOK: train {len(train)}, holdout {len(hold)}, skipped {n_skip}",
          file=sys.stderr)
    print(f"  wrote {OUT}/", file=sys.stderr)


if __name__ == "__main__":
    main()
