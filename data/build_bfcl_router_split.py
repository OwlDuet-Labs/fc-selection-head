#!/usr/bin/env python3
"""Build a STAR selector train/holdout split from a BFCL live category.

Reshapes a BFCL `live_*` category into the selection-head trainer's instance
format, with STAR teacher cosines on each candidate. This is the publishable
SQRL experiment's data: train a STAR DoRA selector on BFCL's OWN tasks and beat
vanilla 0.6B on BFCL's scorer. Originally `live_multiple`-only; `--category`
generalizes it to the categories below — this closes Paper 1 Limitation (c)
("single task family"), giving an across-BFCL-categories comparison.

Categories (`--category`):
  live_multiple  (default) — N candidate functions, gold always among them.
  live_simple    — exactly ONE candidate function (K=1), gold == that function.
                   Same gold-extraction path as live_multiple (a possible_answer
                   file with single-call ground_truth). The K=1 floor makes this
                   the easy end of the selection-difficulty axis.
  live_relevance — ABSTENTION (positive): some function DOES apply; the correct
                   behaviour is to call one (not abstain). No possible_answer
                   file and no per-row named gold — BFCL scores any valid call.
  live_irrelevance — ABSTENTION (negative): NO function applies; the correct
                   behaviour is to abstain. No possible_answer file / named gold.

Instance shape (one JSONL line):
  {prompt, candidates:[{id, desc, star_sim}], gold, gold_position(1-based),
   K, source, abstain(bool)}
  - id        = the BFCL function name (what the selector emits/picks), or the
                sentinel "NONE" for the abstention candidate (see below).
  - desc      = the function description (selector candidate context + teacher
                text); the NONE sentinel carries a fixed abstain description.
  - star_sim  = cos(emb(cand.desc), emb(gold.desc)) — the STAR graded teacher,
                built from the SAME frozen embedder, via shared/teacher_targets.

ABSTENTION CONVENTION (relevance / irrelevance) — load-bearing:
  The selector head emits a 1-based INDEX, so "no tool applies" needs an index
  to point at. We append an explicit sentinel candidate `NONE` ("No function
  applies — abstain.") as the LAST candidate of every relevance/irrelevance
  instance. Then:
    * live_irrelevance (no function applies)  → gold = NONE → gold_position = K
      (the appended sentinel; K already includes it). abstain = True.
    * live_relevance   (some function applies)→ gold = "any real function", which
      BFCL scores as "called something, did NOT abstain". There is no single
      named gold index, so we set gold_position = 1 by convention (a real
      candidate) and mark abstain = False. The category-aware scorer in
      eval/bfcl/run_category_comparison.py grades relevance as "pred != NONE
      index" rather than exact-match — see that file. Exact-match eval
      (router_eval.py / heads.router.eval) therefore UNDER-reports relevance and
      should not be used as the relevance metric; use the category runner.
  The NONE sentinel's teacher cosine is 1.0 with itself (irrelevance gold) and
  ~0 against real candidates, so the STAR term pushes the abstain representation
  away from the function descriptions — the desired geometry.

Split: deterministic by entry-id hash → train ~85% / holdout ~15%. The holdout
holds out ENTRIES (same function vocabulary), the standard in-distribution test.

Teacher uses the function DESCRIPTIONS (not names): two functions are "near" if
their descriptions are semantically close — exactly the discrimination signal.

Usage:
  python build_bfcl_router_split.py [--category live_multiple] \
      [--embedder-dir DIR] [--holdout-frac 0.15]
Writes data/bfcl_<category>/{train,holdout}.jsonl + manifest.json
(live_multiple keeps its historical output dir data/bfcl_router/).
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "shared"))
import teacher_targets as TT

# Point BFCL_DATA at your local BFCL source (the public gorilla/BFCL repo); the
# loader reads <BFCL_DATA>/data/BFCL_v4_<category>.json (+ possible_answer/ for
# the gold-bearing categories).
import os
BFCL = Path(os.environ.get("BFCL_DATA", "/path/to/BFCL"))
DATA_ROOT = Path(__file__).resolve().parent
# A local sentence-embedder for the STAR graded teacher (any HF/MLX sentence embedder);
# set EMBEDDER_DIR. The committed split already includes precomputed teacher targets, so
# this is only needed to rebuild the split from raw BFCL.
DEFAULT_EMBEDDER = Path(os.environ.get("EMBEDDER_DIR", "/path/to/sentence-embedder"))

# Per-category metadata. `gold_kind`:
#   "named"   -> gold is a real function read from possible_answer/ (live_*).
#   "abstain" -> no named gold; gold is the appended NONE sentinel (irrelevance)
#                or "any real candidate" (relevance). See ABSTENTION CONVENTION.
CATEGORIES = {
    "live_multiple": {"gold_kind": "named", "out": "bfcl_router"},
    "live_simple": {"gold_kind": "named", "out": "bfcl_live_simple"},
    "live_relevance": {"gold_kind": "abstain", "out": "bfcl_live_relevance"},
    "live_irrelevance": {"gold_kind": "abstain", "out": "bfcl_live_irrelevance"},
}

# The abstain sentinel appended to every relevance/irrelevance instance.
NONE_ID = "NONE"
NONE_DESC = "No function applies — abstain (do not call any function)."

# Prompt-template drift guard. The selection-head trainer treats
# manifest['template_sha8'] as the lock against train/batch.py template drift;
# this is the current trainer value (re-derive if render_messages changes). All
# categories share the SAME template, so they share this sha.
TEMPLATE_SHA8 = "58cc417d"


def make_embed_fn(embedder_dir):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(embedder_dir), device="mps")

    def embed(texts):
        import numpy as np
        v = model.encode(list(texts), batch_size=64, show_progress_bar=False,
                         convert_to_numpy=True, normalize_embeddings=True)
        return v.astype("float32")
    return embed


def load_category(category: str):
    """(entries, gold_truth_by_id) for a category.

    `gold_truth_by_id` is {} for abstention categories (no possible_answer file).
    """
    data_file = BFCL / "data" / f"BFCL_v4_{category}.json"
    entries = [json.loads(l) for l in open(data_file)]
    gt = {}
    gt_file = BFCL / "data" / "possible_answer" / f"BFCL_v4_{category}.json"
    if gt_file.exists():
        gt = {g["id"]: g for g in
              (json.loads(l) for l in open(gt_file))}
    return entries, gt


def build_instances(entries, gt, gold_kind, embed, category):
    """(instances, n_skip). One instance per usable entry, with STAR cosines.

    Gold-bearing ("named") categories follow the original live_multiple path:
    gold = first key of ground_truth[0], must appear among the entry's functions.
    Abstention categories append the NONE sentinel and route gold per the
    ABSTENTION CONVENTION in the module docstring.
    """
    # 1. collect every (fn name -> description) across the corpus for the teacher.
    fn_desc = {}
    for e in entries:
        for f in e["function"]:
            fn_desc.setdefault(f["name"], f.get("description", "") or f["name"])
    if gold_kind == "abstain":
        fn_desc[NONE_ID] = NONE_DESC  # sentinel participates in the teacher space
    print(f"  {len(entries)} entries, {len(fn_desc)} distinct functions",
          file=sys.stderr)

    # 2. embed every unique function description ONCE → teacher vectors.
    key2vec, sha = TT.build_text_vectors(fn_desc, embed)
    print(f"  embedded {len(key2vec)} fn descriptions (sha8={sha})", file=sys.stderr)

    # 3. build instances with STAR teacher cosines per candidate.
    instances = []
    n_skip = 0
    for e in entries:
        cand_names = [f["name"] for f in e["function"]]

        if gold_kind == "named":
            if e["id"] not in gt:
                n_skip += 1
                continue
            gold = list(gt[e["id"]]["ground_truth"][0].keys())[0]
            if gold not in cand_names:
                n_skip += 1
                continue
            gold_pos = cand_names.index(gold) + 1  # 1-based
            abstain = False
        else:
            # Abstention: append the NONE sentinel as the last candidate.
            cand_names = cand_names + [NONE_ID]
            if category == "live_irrelevance":
                # No function applies → gold IS the sentinel (last index).
                gold = NONE_ID
                gold_pos = len(cand_names)  # 1-based, the appended sentinel
                abstain = True
            else:  # live_relevance
                # Some function applies → "call a real one". No named gold index;
                # convention = position 1 (a real candidate). The category-aware
                # scorer grades this as pred != NONE, not exact-match.
                if len(cand_names) <= 1:  # only the sentinel; nothing to call
                    n_skip += 1
                    continue
                gold = cand_names[0]
                gold_pos = 1
                abstain = False

        sims = TT.teacher_for_instance(cand_names, gold, key2vec)  # cos to gold
        cands = [{"id": n,
                  "desc": (NONE_DESC if n == NONE_ID else fn_desc[n])[:120],
                  "star_sim": (sims[i] if sims else (1.0 if n == gold else 0.0))}
                 for i, n in enumerate(cand_names)]
        inst = {"prompt": e["question"][0][0]["content"], "candidates": cands,
                "gold": gold, "gold_position": gold_pos, "K": len(cands),
                "source": f"bfcl_{category}",
                # carried only for the deterministic split; stripped before write.
                "_entry_id": e["id"]}
        # `abstain` is meaningful only for abstention categories; emitting it only
        # there keeps live_multiple/live_simple instances clean (and reproduces
        # the committed bfcl_router split byte-for-byte).
        if gold_kind == "abstain":
            inst["abstain"] = abstain
        instances.append(inst)

    return instances, n_skip, fn_desc, sha


def split_instances(instances, holdout_frac):
    """Deterministic train/holdout split by BFCL entry-id hash.

    Identical hashing to the original live_multiple builder (hash the BFCL entry
    id), so rebuilding `live_multiple` reproduces the committed bfcl_router split
    byte-for-byte. The transient `_entry_id` key is stripped before writing.
    """
    train, hold = [], []
    for inst in instances:
        h = int(hashlib.sha256(str(inst["_entry_id"]).encode()).hexdigest(), 16) % 1000
        out = inst.copy()
        out.pop("_entry_id", None)
        (hold if h < holdout_frac * 1000 else train).append(out)
    return train, hold


def main():
    ap = argparse.ArgumentParser(
        description="Build a STAR selector split from a BFCL live category.")
    ap.add_argument("--category", choices=list(CATEGORIES), default="live_multiple",
                    help="BFCL category to build (default: live_multiple).")
    ap.add_argument("--embedder-dir", type=Path, default=DEFAULT_EMBEDDER)
    ap.add_argument("--holdout-frac", type=float, default=0.15)
    ap.add_argument("--star-temp", type=float, default=0.08)
    args = ap.parse_args()

    meta = CATEGORIES[args.category]
    out = DATA_ROOT / meta["out"]

    entries, gt = load_category(args.category)
    embed = make_embed_fn(args.embedder_dir)
    instances, n_skip, fn_desc, sha = build_instances(
        entries, gt, meta["gold_kind"], embed, args.category)
    train, hold = split_instances(instances, args.holdout_frac)

    out.mkdir(parents=True, exist_ok=True)
    (out / "train.jsonl").write_text(
        "\n".join(json.dumps(x) for x in train) + "\n")
    (out / "holdout.jsonl").write_text(
        "\n".join(json.dumps(x) for x in hold) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "source": f"BFCL_v4_{args.category}", "category": args.category,
        "gold_kind": meta["gold_kind"],
        "n_train": len(train), "n_holdout": len(hold), "n_skip": n_skip,
        "n_functions": len(fn_desc), "fn_desc_sha8": sha,
        "star_temp": args.star_temp, "holdout_frac": args.holdout_frac,
        # the selection-head trainer asserts manifest['template_sha8'] ==
        # train_selection_head.template_sha8() (prompt-template drift guard).
        # This is the current trainer value; re-derive if the template changes.
        "template_sha8": TEMPLATE_SHA8}, indent=2))
    print(f"\nOK [{args.category}]: train {len(train)}, holdout {len(hold)}, "
          f"skipped {n_skip}", file=sys.stderr)
    print(f"  wrote {out}/", file=sys.stderr)


if __name__ == "__main__":
    main()
