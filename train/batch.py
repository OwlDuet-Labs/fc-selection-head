"""Head-agnostic tokenization + candidate/query span location for the trainer.

Candidate-list rendering + index-target batching for the selector head.
(`render_messages` / `tokenize_for_loss` / `locate_candidate_spans` /
`locate_query_span` / `make_batch`) and STRIPPED to the clean index-target
selection path.
  * TYPED_CONTEXT / typed_context_lib (production typedTags surface)
  * STRING_TARGET (api-name target variant)
  * SHUFFLE_CANDIDATES candidate-order augmentation
  * hard-negative mining metadata (idx_logit_pos, neg digit/span, online pool)
  * the template_sha8 drift guard

What remains is the geometry the STAR loss needs:
  * the `[N] candidate_id` numbered-candidate template,
  * prompt-loss masking (loss only on the assistant `{"index": N}` tokens),
  * deterministic per-candidate id-token spans + the query span (for the
    contrastive pooled representations consumed by shared/star_loss).

The selection template is INDEX-target: the assistant emits `{"index": N}`. The
CE signal lives on the first digit token of N; the STAR contrastive signal lives
on the pooled candidate/query hidden spans this module locates.

Pure tokenizer + MLX padding; no model, no autograd. Spans are located ONCE at
batch-build time (string round-trips), never inside the value_and_grad loop.
"""
from __future__ import annotations

import mlx.core as mx

# --- the index-target selection template ----------
TEMPLATE_SYS = (
    "You are a routing assistant. Given a USER REQUEST and a numbered list "
    "of CANDIDATES (plans/tools), choose the single best candidate for the "
    "request and respond with ONLY a JSON object of the form "
    '{"index": N} where N is the 1-based candidate index.'
)
TEMPLATE_USER_FMT = (
    "USER REQUEST: {prompt}\n\n"
    "CANDIDATES ({K}):\n{cand_block}\n\n"
    "Respond with one JSON object only."
)
TEMPLATE_ASSIST_FMT = '{{"index": {idx}}}'

# Marker-token cache (resolved once per tokenizer).
_DIGIT_TOK: dict[int, int] | None = None
_TOK_LBRACK: int | None = None
_TOK_RBRACK: int | None = None
_TOK_NL: int | None = None
_TOK_NL2: int | None = None


def _build_digit_tok(tokenizer) -> dict[int, int]:
    global _DIGIT_TOK
    if _DIGIT_TOK is not None:
        return _DIGIT_TOK
    d = {}
    for n in range(10):
        ids = tokenizer.encode(str(n), add_special_tokens=False)
        assert len(ids) == 1, f"digit {n} not single-token: {ids}"
        d[n] = ids[0]
    _DIGIT_TOK = d
    return d


def first_digit_token(tokenizer, pos_1based: int) -> int:
    """First-digit token id for a 1-based candidate index (the CE target)."""
    d = _build_digit_tok(tokenizer)
    return d[int(str(pos_1based)[0])]


def _build_marker_toks(tokenizer) -> tuple[int, int, set]:
    """Resolve ' [', ']', newline marker token ids (single-token; cached)."""
    global _TOK_LBRACK, _TOK_RBRACK, _TOK_NL, _TOK_NL2
    if _TOK_LBRACK is not None:
        return _TOK_LBRACK, _TOK_RBRACK, {_TOK_NL, _TOK_NL2}
    lb = tokenizer.encode(" [", add_special_tokens=False)
    rb = tokenizer.encode("]", add_special_tokens=False)
    nl = tokenizer.encode("\n", add_special_tokens=False)
    nl2 = tokenizer.encode("\n\n", add_special_tokens=False)
    assert len(lb) == 1 and len(rb) == 1 and len(nl) == 1 and len(nl2) == 1, (
        f"marker tokens not single-token: ' ['={lb} ']'={rb} "
        f"'\\n'={nl} '\\n\\n'={nl2}"
    )
    _TOK_LBRACK, _TOK_RBRACK, _TOK_NL, _TOK_NL2 = lb[0], rb[0], nl[0], nl2[0]
    return _TOK_LBRACK, _TOK_RBRACK, {_TOK_NL, _TOK_NL2}


def render_messages(prompt: str, candidates: list[dict], gold_index_1b: int):
    """system + user + assistant chat messages for one selection instance."""
    cand_block = "\n".join(
        f"  [{i + 1}] {c['id']}" for i, c in enumerate(candidates)
    )
    user = TEMPLATE_USER_FMT.format(
        prompt=prompt, K=len(candidates), cand_block=cand_block
    )
    assist = TEMPLATE_ASSIST_FMT.format(idx=gold_index_1b)
    return [
        {"role": "system", "content": TEMPLATE_SYS},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assist},
    ]


def locate_candidate_spans(
    tokenizer, full_ids: list[int], n_cands: int, prefix_len: int
) -> list[tuple[int, int]]:
    """Per-candidate id-token (start, end) spans, full_ids coordinate.

    Scans the prefix for ` [`<digits>`]`<id-run><newline>. Returns one (start,
    end) per 1-based candidate (end exclusive); (-1, -1) if not locatable.
    Defensive: never raises.
    """
    lbrack, rbrack, newline_toks = _build_marker_toks(tokenizer)
    digit_ids = set(_build_digit_tok(tokenizer).values())
    spans: dict[int, tuple[int, int]] = {}
    j = 0
    hi = min(prefix_len, len(full_ids))
    while j < hi:
        if int(full_ids[j]) != lbrack:
            j += 1
            continue
        k = j + 1
        digits = ""
        while k < hi and int(full_ids[k]) in digit_ids:
            digits += tokenizer.decode([int(full_ids[k])])
            k += 1
        if k >= hi or int(full_ids[k]) != rbrack or not digits.isdigit():
            j += 1
            continue
        cand_idx = int(digits)  # 1-based
        start = k + 1
        e = start
        while e < hi:
            t = int(full_ids[e])
            if t in newline_toks or t == lbrack:
                break
            e += 1
        if 1 <= cand_idx <= n_cands and start < e:
            spans.setdefault(cand_idx, (start, e))
        j = e
    return [spans.get(i, (-1, -1)) for i in range(1, n_cands + 1)]


def locate_query_span(
    tokenizer, full_ids: list[int], prefix_len: int
) -> tuple[int, int]:
    """(start, end) token span of the user prompt, full_ids coordinate.

    The prompt sits between "USER REQUEST: " and "\\n\\nCANDIDATES" in the user
    template. Maps those anchors to token offsets by an incremental decode.
    Returns (-1, -1) if anchors are missing. `end` is exclusive.
    """
    hi = min(prefix_len, len(full_ids))
    cum = []
    acc = ""
    for t in full_ids[:hi]:
        acc = acc + tokenizer.decode([int(t)])
        cum.append(len(acc))
    dec = acc
    cs = dec.find("USER REQUEST: ")
    if cs < 0:
        return (-1, -1)
    pstart = cs + len("USER REQUEST: ")
    ce = dec.find("\n\nCANDIDATES", pstart)
    if ce < 0:
        return (-1, -1)

    def char_to_tok(charoff: int) -> int:
        for ti, cl in enumerate(cum):
            if cl > charoff:
                return ti
        return len(cum)

    qs = char_to_tok(pstart)
    qe = char_to_tok(ce)
    # Trailing prompt punctuation can merge with "\n\n" into one straddling
    # token; include it so the span is an exact round-trip of the prompt.
    if qe < len(cum):
        tok_start = cum[qe - 1] if qe > 0 else 0
        if tok_start < ce:
            qe += 1
    if qs >= qe:
        return (-1, -1)
    return (qs, qe)


def tokenize_for_loss(tokenizer, instance: dict):
    """(input_ids[L], loss_mask[L-1], meta) for one selection instance.

    loss_mask is 1 only over the assistant `{"index": N}` tokens (prompt-loss
    masking). meta carries the per-candidate id spans, the query span, the gold
    1-based position, and the loss-array position predicting the gold first
    digit (CE target). Spans are full_ids coordinate; the loss array is
    full_ids[1:], so a span position p maps to loss index p-1.
    """
    gold_idx = instance["gold_position"]
    prompt = instance["prompt"]
    candidates = instance["candidates"]

    full_msgs = render_messages(prompt, candidates, gold_idx)
    prefix_msgs = full_msgs[:-1]  # system + user
    # enable_thinking=False to match production + the eval surface: Qwen3 would
    # otherwise emit a <think> block before the answer (the digit then never
    # appears in the assistant target / early generation). The template injects
    # an empty <think>\n\n</think> and the answer follows directly.
    prefix_text = tokenizer.apply_chat_template(
        prefix_msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False,
        enable_thinking=False,
    )
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    prefix_len = len(prefix_ids)

    # Loss mask: 0 over the prefix, 1 over the assistant continuation. Aligns to
    # full_ids[1:] (token i predicts token i+1).
    L = len(full_ids)
    loss_mask = [0.0] * (L - 1)
    for i in range(prefix_len - 1, L - 1):
        if 0 <= i < L - 1:
            loss_mask[i] = 1.0

    cand_spans = locate_candidate_spans(tokenizer, full_ids, len(candidates), prefix_len)
    query_span = locate_query_span(tokenizer, full_ids, prefix_len)

    # CE target position: the loss-array index that predicts the gold idx's
    # first digit. The assistant block is '{"index": N}', so the first digit
    # token sits at the first assistant position whose token is a digit.
    digit_ids = set(_build_digit_tok(tokenizer).values())
    idx_logit_pos = -1
    for i in range(prefix_len, L):
        if int(full_ids[i]) in digit_ids:
            idx_logit_pos = i - 1  # loss-array coordinate
            break

    meta = {
        "cand_spans": cand_spans,
        "query_span": query_span,
        "gold_position": gold_idx,            # 1-based
        "idx_logit_pos": idx_logit_pos,
        "gold_digit_tok": first_digit_token(tokenizer, gold_idx),
        "star_sims": [c.get("star_sim", 0.0) for c in candidates],
    }
    return mx.array(full_ids, dtype=mx.int32), mx.array(loss_mask, dtype=mx.float32), meta


def make_batch(tokenizer, batch_insts: list[dict]) -> dict:
    """Pad-batched tensors + per-candidate spans for CE + STAR.

    Returns:
      inputs    int32   [B, Lmax]
      targets   int32   [B, Lmax-1]   (inputs shifted)
      mask      float32 [B, Lmax-1]   prompt-loss mask
      idx_pos   int32   [B]           loss-array pos predicting gold first digit
      gold_tok  int32   [B]           gold first-digit token id (CE target value)
      cand_spans int32  [B, Kmax, 2]  per-candidate id span (loss-array coord; -1 absent)
      query_span int32  [B, 2]        query span (loss-array coord; -1 absent)
      gold_idx0 int32   [B]           0-based gold candidate index
      k_valid   float32 [B, Kmax]     1.0 where candidate k is present
      star_tgt  float32 [B, Kmax]     teacher cosines (star_sim) per candidate
      q_valid   float32 [B]           1.0 where the query span is valid
    """
    toks, masks, metas = [], [], []
    for inst in batch_insts:
        ids, m, meta = tokenize_for_loss(tokenizer, inst)
        toks.append(ids)
        masks.append(m)
        metas.append(meta)

    B = len(toks)
    Lmax = max(int(t.shape[0]) for t in toks)
    Kmax = max(len(m["cand_spans"]) for m in metas)
    pad_id = tokenizer.eos_token_id or 0

    inputs = mx.full((B, Lmax), pad_id, dtype=mx.int32)
    mask = mx.zeros((B, Lmax - 1), dtype=mx.float32)
    idx_pos = [-1] * B
    gold_tok = [0] * B
    cand_spans = [[(-1, -1)] * Kmax for _ in range(B)]
    query_span = [(-1, -1)] * B
    gold_idx0 = [0] * B
    k_valid = [[0.0] * Kmax for _ in range(B)]
    star_tgt = [[0.0] * Kmax for _ in range(B)]
    q_valid = [0.0] * B

    # Span coords are full_ids coordinate; the loss array is full_ids[1:], so a
    # full_ids position p maps to loss index p-1. Clamp into [0, Lmax-2].
    def _to_loss_span(se):
        s, e = se
        if s < 0 or e <= s:
            return (-1, -1)
        ls, le = s - 1, e - 1
        ls = max(0, min(ls, Lmax - 2))
        le = max(ls + 1, min(le, Lmax - 1))
        return (ls, le)

    rows_in = []
    rows_m = []
    for b, (ids, m, meta) in enumerate(zip(toks, masks, metas)):
        L = int(ids.shape[0])
        row = mx.concatenate([ids, mx.full((Lmax - L,), pad_id, dtype=mx.int32)])
        rows_in.append(row[None, :])
        mrow = mx.concatenate([m, mx.zeros((Lmax - 1 - int(m.shape[0]),), dtype=mx.float32)])
        rows_m.append(mrow[None, :])

        idx_pos[b] = int(meta["idx_logit_pos"])
        gold_tok[b] = int(meta["gold_digit_tok"])
        gold_idx0[b] = int(meta["gold_position"]) - 1
        qs = _to_loss_span(meta["query_span"])
        query_span[b] = qs
        q_valid[b] = 1.0 if qs[0] >= 0 else 0.0
        for k, se in enumerate(meta["cand_spans"]):
            ls = _to_loss_span(se)
            cand_spans[b][k] = ls
            if ls[0] >= 0:
                k_valid[b][k] = 1.0
            star_tgt[b][k] = float(meta["star_sims"][k]) if k < len(meta["star_sims"]) else 0.0

    inputs = mx.concatenate(rows_in, axis=0)
    mask = mx.concatenate(rows_m, axis=0)
    targets = inputs[:, 1:]

    return {
        "inputs": inputs,
        "targets": targets,
        "mask": mask,
        "idx_pos": mx.array(idx_pos, dtype=mx.int32),
        "gold_tok": mx.array(gold_tok, dtype=mx.int32),
        "cand_spans": mx.array(cand_spans, dtype=mx.int32),     # [B,Kmax,2]
        "query_span": mx.array(query_span, dtype=mx.int32),     # [B,2]
        "gold_idx0": mx.array(gold_idx0, dtype=mx.int32),
        "k_valid": mx.array(k_valid, dtype=mx.float32),
        "star_tgt": mx.array(star_tgt, dtype=mx.float32),
        "q_valid": mx.array(q_valid, dtype=mx.float32),
    }
