"""STAR — Similarity-guided soft-target KD loss for SQRL heads.

The graded soft-contrastive (SCE-style) target and one-hot InfoNCE loss (the
`hn_loss == "star"` branch + `_pool_span`/`_l2norm`), generalized into a
standalone, head-agnostic module. The router trainer was the first consumer; the
slot-fill value-span head is the second. See `docs/STAR.md`.

THE IDEA. A small model's selection head, trained with a ONE-HOT positive
(standard InfoNCE / hard-negative contrastive), MEMORIZES a handful of pinned
confusion pairs — holdout accuracy climbs while honest e2e drops. STAR replaces
the one-hot target with a GRADED teacher distribution over the candidate set,
built from the retrieval embedder's own geometry:

    p_teacher_k = softmax( cos(emb(cand_k), emb(gold)) / star_temp )   over k

The loss is the soft cross-entropy H(p_teacher, p_model). The gold (cos≈1 with
itself) is the peak; near-twins get graded mass; unrelated candidates ≈0. The
head learns the *shape* of the candidate neighbourhood rather than pushing one
fixed pair apart — so it generalizes off the pinned set. On an internal product
router this was the first net-positive retrain where every binary-signal lever
memorized.

INVARIANTS (carried verbatim; the trainers depend on them):
  * `lambda == 0` ⇒ the caller adds ZERO STAR term (bit-identical to stock CE).
    This module computes only the STAR term; the caller is responsible for the
    `ce + lambda * star_term` combination and the lambda<=0 short-circuit.
  * one-hot teacher ⇒ STAR ≡ InfoNCE (proven by the trainer's unit test).
  * invalid/padding candidates carry ZERO teacher mass and are masked out of the
    student log-prob sum — no NaN.

All math is MLX (`mlx.core`). No model / no I/O — pure tensor ops, so it unit-
tests on tiny synthetic tensors with no MLX device.
"""
from __future__ import annotations

import mlx.core as mx

NEG_INF = mx.array(-1e9)


def pool_span(h, span_se, hidden):
    """Mask-safe MEAN-pool of hidden states over a per-instance (start,end) span.

    h:        [B, T, H] hidden states (loss-array coordinate).
    span_se:  [B, 2] int32 (start,end); (-1,*) marks an absent span.
    hidden:   int H (kept for signature parity / a broadcast guard).

    Returns (pooled [B,H], valid [B] float). An absent/empty span pools to
    all-zeros with valid=0 (caller masks it — no NaN). Differentiable w.r.t. h
    and never indexes out of range (built from a per-position membership mask).
    """
    B, T, H = h.shape
    start = span_se[:, 0]
    end = span_se[:, 1]
    valid = (start >= 0).astype(h.dtype)
    s_safe = mx.maximum(start, mx.array(0))
    e_safe = mx.maximum(end, mx.array(0))
    pos = mx.arange(T)[None, :]
    span_mask = ((pos >= s_safe[:, None]) & (pos < e_safe[:, None])).astype(h.dtype)
    span_mask = span_mask * valid[:, None]
    counts = span_mask.sum(axis=-1)
    counts_safe = mx.maximum(counts, mx.array(1.0))
    pooled = (h * span_mask[:, :, None]).sum(axis=1) / counts_safe[:, None]
    valid = valid * (counts > 0).astype(h.dtype)
    return pooled, valid


def l2norm(x, eps=1e-6):
    """L2-normalize the last axis (guarded against zero rows)."""
    n = mx.sqrt((x * x).sum(axis=-1, keepdims=True) + eps)
    return x / n


def candidate_sims(h_query, h_cand, k_valid, tau):
    """cos(query, cand_k)/tau over the candidate set, masking invalids to -inf.

    h_query:  [B, H]   L2-normalized query representation.
    h_cand:   [B, K, H] L2-normalized candidate representations.
    k_valid:  [B, K]   1.0 where candidate k is a real (present) candidate.
    Returns sims [B,K] (invalid → -1e9) and logZ [B] = logsumexp over valid.
    """
    sims = (h_cand * h_query[:, None, :]).sum(axis=-1) / tau     # [B,K]
    sims = mx.where(k_valid > 0, sims, NEG_INF)
    logZ = mx.logsumexp(sims, axis=-1)                           # [B]
    return sims, logZ


def star_term(sims, logZ, k_valid, star_target, gold_idx0, q_valid,
              star_temp=0.08):
    """The STAR soft cross-entropy term H(p_teacher, p_model) over candidates.

    sims, logZ:   from `candidate_sims` ([B,K] / [B]).
    k_valid:      [B,K] candidate-present mask.
    star_target:  [B,K] teacher cosines cos(emb(cand_k), emb(gold)); built by
                  the teacher-target builder from the FROZEN retrieval embedder.
    gold_idx0:    [B] 0-based gold candidate index (for the usability mask).
    q_valid:      [B] 1.0 where the query representation is valid.
    star_temp:    teacher softmax temperature (winning router value: 0.08).

    Returns a scalar = mean STAR cross-entropy over USABLE instances. An instance
    is usable iff the query is valid, the gold candidate is present, AND at least
    one OTHER candidate is a valid negative (else the soft target is degenerate).
    Caller combines as `ce + lambda * star_term(...)`.
    """
    K = sims.shape[1]
    gold_safe = mx.maximum(gold_idx0, mx.array(0)).astype(mx.int32)
    gold_onehot = (mx.arange(K)[None, :] == gold_idx0[:, None]).astype(mx.float32)
    n_neg_valid = (k_valid * (1.0 - gold_onehot)).sum(axis=-1)              # [B]
    gold_valid = mx.take_along_axis(k_valid, gold_safe[:, None], axis=-1).squeeze(-1)
    use = q_valid * gold_valid * (n_neg_valid > 0).astype(mx.float32)       # [B]

    # Teacher distribution over VALID candidates (gold is the peak; absent/
    # padding candidates get -inf → zero mass).
    star_T = mx.array(star_temp)
    t_logits = mx.where(k_valid > 0, star_target / star_T, NEG_INF)         # [B,K]
    t_logZ = mx.logsumexp(t_logits, axis=-1, keepdims=True)                 # [B,1]
    p_teacher = mx.exp(t_logits - t_logZ)                                   # [B,K]

    # Student log-probs over the SAME valid candidate set.
    log_p_model = sims - logZ[:, None]                                      # [B,K]

    # Soft cross-entropy -Σ_k p_teacher_k · log_p_model_k; invalid k have
    # p_teacher=0 so their -inf log_p is masked to 0 (no NaN).
    star_ce_k = mx.where(k_valid > 0, -p_teacher * log_p_model, 0.0)        # [B,K]
    star_ce = star_ce_k.sum(axis=-1) * use                                 # [B]
    denom = mx.maximum(use.sum(), mx.array(1.0))
    return star_ce.sum() / denom
