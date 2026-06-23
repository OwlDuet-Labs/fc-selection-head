"""Router head — selection over K retrieved candidates (CE + STAR teacher KD).

The first SQRL head ported to the self-contained trainer. Implements the Head
protocol (docs/PORT_PLAN.md §4): load_data / make_batch / loss / eval /
render_inference. The loss is the methods-paper winner: prompt-masked CE on the
`{"index": N}` target PLUS the STAR graded-teacher contrastive term over pooled
candidate hidden spans, combined as `ce + lambda * star_term`.

All the loss MATH lives in shared/star_loss (pool_span, candidate_sims,
star_term) and shared/base (hidden_states, logits) — this module is the
orchestration: tokenize via train/batch, pool spans, call the shared terms,
score holdout selection accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

from shared import star_loss
from train.batch import make_batch, render_messages

NAME = "router"


def load_data(data_dir: str):
    """(train, holdout) instance lists from data/bfcl_router/{train,holdout}.jsonl."""
    d = Path(data_dir)
    train = [json.loads(l) for l in open(d / "train.jsonl")]
    holdout = [json.loads(l) for l in open(d / "holdout.jsonl")]
    return train, holdout


def build_batch(tokenizer, insts):
    return make_batch(tokenizer, insts)


def loss(model, batch, cfg):
    """ce + star_lambda * star_term. lambda<=0 ⇒ bit-identical pure CE."""
    inputs = batch["inputs"]
    targets = batch["targets"]
    mask = batch["mask"]

    logits = model(inputs)[:, :-1, :].astype(mx.float32)
    log_probs = nn.log_softmax(logits, axis=-1)
    nll = -mx.take_along_axis(log_probs, targets[..., None], axis=-1).squeeze(-1)
    nll = nll * mask
    ce = nll.sum() / mx.maximum(mask.sum(), mx.array(1.0))

    if cfg.star_lambda <= 0.0:
        return ce

    # Pre-lm_head hidden states, pooled per candidate/query span → STAR term.
    h = model.model(inputs)[:, :-1, :].astype(mx.float32)
    H = h.shape[-1]
    h_query, q_valid = star_loss.pool_span(h, batch["query_span"], H)
    h_query = star_loss.l2norm(h_query)

    cand_spans = batch["cand_spans"]   # [B,K,2]
    K = cand_spans.shape[1]
    pooled, pvalid = [], []
    for k in range(K):
        hk, vk = star_loss.pool_span(h, cand_spans[:, k, :], H)
        pooled.append(star_loss.l2norm(hk))
        pvalid.append(vk)
    h_cand = mx.stack(pooled, axis=1)                      # [B,K,H]
    k_valid = mx.stack(pvalid, axis=1) * batch["k_valid"]  # [B,K]

    sims, logZ = star_loss.candidate_sims(h_query, h_cand, k_valid, tau=cfg.star_temp)
    term = star_loss.star_term(
        sims, logZ, k_valid, batch["star_tgt"], batch["gold_idx0"],
        q_valid, star_temp=cfg.star_temp,
    )
    return ce + cfg.star_lambda * term


def render_inference(tokenizer, inst) -> str:
    """Prompt text for greedy selection (system+user, generation-primed)."""
    msgs = render_messages(inst["prompt"], inst["candidates"], inst["gold_position"])[:-1]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )


def _predict_index(model, tokenizer, inst) -> int:
    """Greedy 1-based index the router selects for one instance (-1 on parse fail)."""
    text = render_inference(tokenizer, inst)
    ids = mx.array(tokenizer.encode(text, add_special_tokens=False), dtype=mx.int32)[None, :]
    # Generate a few tokens; the index digit is the first digit after '{"index":'.
    digit_ids = {tokenizer.encode(str(n), add_special_tokens=False)[0]: n for n in range(10)}
    cur = ids
    for _ in range(8):
        logit = model(cur)[:, -1, :]
        nxt = int(mx.argmax(logit, axis=-1).item())
        if nxt in digit_ids:
            return digit_ids[nxt]
        cur = mx.concatenate([cur, mx.array([[nxt]], dtype=mx.int32)], axis=1)
    return -1


def eval(model, tokenizer, holdout, max_n: int = 200) -> float:
    """Selection accuracy: fraction where the greedy index == gold_position."""
    n = min(len(holdout), max_n)
    correct = 0
    for inst in holdout[:n]:
        pred = _predict_index(model, tokenizer, inst)
        if pred == int(inst["gold_position"]):
            correct += 1
        mx.clear_cache()   # eval over many instances accumulates buffer cache too
    return correct / max(n, 1)
