"""STAR teacher-target builder — corpus-agnostic.

Builds the per-candidate teacher cosines `cos(emb(cand_k), emb(gold))` that
`shared.star_loss.star_term` consumes as its graded soft target. Lifted from
Builds the graded teacher's candidate↔gold cosine targets. The
original hardcoded `R.build_corpus(plan_db, tools_db)`; this version takes
already-resolved candidate/gold *text* and a generic embedder, so the SAME
builder serves the BFCL split (candidate text = the rendered candidate the
model sees at inference), embedded with the frozen retrieval embedder. The
`embed_text`/alias builder; for BFCL it's the same embedder over the function
descriptions BFCL hands the model. Callers pass a `embed_fn` so this module never
pins a particular embedder.

Efficiency: embed each UNIQUE candidate/gold text ONCE (a few hundred strings),
then every per-instance cosine is a cached dot product of L2-normalized vectors.

Pure numpy + an injected `embed_fn(list[str]) -> np.ndarray[N,D] (L2-normalized)`.
No model is imported here — the caller supplies the embedder (frozen Gemma for
the served/loaded embedder for BFCL). Unit-testable with a fake
`embed_fn`.
"""
from __future__ import annotations

import hashlib
from typing import Callable, Sequence

import numpy as np

EmbedFn = Callable[[Sequence[str]], np.ndarray]  # (texts) -> [N,D] L2-normalized


def sha8(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


def build_text_vectors(text_by_key: dict, embed_fn: EmbedFn):
    """key -> L2-normalized vector, embedding each unique TEXT once.

    text_by_key: {key: embedding_text}. Keys are candidate/gold ids or
    function names (BFCL). Returns (key -> np.float32 vector, sha8 of the
    text manifest for provenance).
    """
    keys = list(text_by_key)
    texts = [text_by_key[k] for k in keys]
    manifest_sha = sha8(" ".join(f"{k}={text_by_key[k]}" for k in sorted(keys)))
    vecs = np.asarray(embed_fn(texts), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"embed_fn must return L2-normalized vectors "
        f"(min={norms.min():.4f} max={norms.max():.4f})"
    )
    return {k: v for k, v in zip(keys, vecs)}, manifest_sha


def teacher_for_instance(candidate_keys, gold_key, key2vec, missing="far"):
    """Graded teacher cosines for ONE instance's candidate list.

    candidate_keys: ordered list of the instance's candidate ids/names (the same
                    order the router sees them; teacher aligns to it).
    gold_key:       the gold candidate's key (must be in candidate_keys).
    key2vec:        from `build_text_vectors`.
    missing:        policy for a candidate/gold absent from key2vec:
                      "far"  -> star_sim 0.0 (treated as an unrelated candidate)
                      "raise"-> assert (use when coverage must be total)

    Returns list[float] star_sim per candidate (gold ≈ 1.0 with itself). When the
    GOLD has no vector, returns None so the trainer back-fills a one-hot (STAR
    degrades to InfoNCE for that instance — never silently wrong).
    """
    gv = key2vec.get(gold_key)
    if gv is None:
        if missing == "raise":
            raise AssertionError(f"gold {gold_key!r} has no vector")
        return None
    out = []
    for ck in candidate_keys:
        cv = key2vec.get(ck)
        if cv is None:
            if missing == "raise":
                raise AssertionError(f"candidate {ck!r} has no vector")
            out.append(0.0)
        else:
            out.append(float(np.dot(cv, gv)))  # cos of unit vectors == dot
    return out


def coverage_report(instances, key2vec, gold_of, candidates_of):
    """(gold_coverage, cand_coverage) over a set of instances — for the fail-loud
    gate (the original refused to write below 0.98 gold coverage, since a one-hot
    teacher = InfoNCE defeats STAR). `gold_of`/`candidates_of` extract keys from
    an instance."""
    n = n_gold_miss = n_cand = n_cand_miss = 0
    for inst in instances:
        n += 1
        if gold_of(inst) not in key2vec:
            n_gold_miss += 1
        for ck in candidates_of(inst):
            n_cand += 1
            if ck not in key2vec:
                n_cand_miss += 1
    gold_cov = 1.0 - (n_gold_miss / n if n else 0.0)
    cand_cov = 1.0 - (n_cand_miss / n_cand if n_cand else 0.0)
    return gold_cov, cand_cov
