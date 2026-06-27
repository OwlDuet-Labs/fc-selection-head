"""Does the trained selector head benefit from MORE candidate context at eval time?

The head was TRAINED on bare names (`[i] {id}`). This evals the actual trained head (0.6B base +
STAR adapter) at three eval-time context levels, keeping its native `{"index": N}` output
protocol and scoring unchanged — only the candidate-block rendering varies:

  names  : `[i] {id}`                       (train distribution; ~100 tok)
  desc   : `[i] {id}: {desc}`               (BFCL terse desc)
  schema : `[i] {id} {full BFCL schema}`    (high context; ~860 tok)

names should reproduce the published 91.0; desc/schema are OFF-DISTRIBUTION (the head never saw
them), so this measures whether extra context helps, hurts, or is flat for our head.

Usage:
  python eval/bfcl/head_context_eval.py \
    --base mlx-community/Qwen3-0.6B-bf16 \
    --adapter adapters/m1-router-star/seed1/adapters.safetensors --label trained-0.6B
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mlx.core as mx
from mlx.utils import tree_unflatten

from shared import base
import train.batch as B

RAW_BFCL = Path("/Volumes/X10/owl/app/daw-mcp/u-factoring-bfcl/.venv/lib/python3.11/"
                "site-packages/bfcl_eval/data/BFCL_v4_live_multiple.json")


def name2schema():
    n = {}
    for line in open(RAW_BFCL):
        for f in json.loads(line)["function"]:
            n[f["name"]] = f
    return n


def cand_block(candidates, n2s, level):
    out = []
    for i, c in enumerate(candidates):
        tag = f"  [{i+1}]"
        if level == "names":
            out.append(f"{tag} {c['id']}")
        elif level == "desc":
            out.append(f"{tag} {c['id']}: {c.get('desc','')}")
        else:
            out.append(f"{tag} {c['id']} {json.dumps(n2s.get(c['id'], {'name': c['id']}))}")
    return "\n".join(out)


def predict_index(model, tok, inst, n2s, level):
    """The head's greedy 1-based index, with candidates rendered at `level`."""
    block = cand_block(inst["candidates"], n2s, level)
    user = B.TEMPLATE_USER_FMT.format(prompt=inst["prompt"], K=len(inst["candidates"]),
                                      cand_block=block)
    msgs = [{"role": "system", "content": B.TEMPLATE_SYS},
            {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = mx.array(tok.encode(text, add_special_tokens=False), dtype=mx.int32)[None, :]
    n_in = ids.shape[1]
    digit_ids = {tok.encode(str(d), add_special_tokens=False)[0]: d for d in range(10)}
    cur = ids
    for _ in range(8):
        nxt = int(mx.argmax(model(cur)[:, -1, :], axis=-1).item())
        if nxt in digit_ids:
            return digit_ids[nxt], n_in
        cur = mx.concatenate([cur, mx.array([[nxt]], dtype=mx.int32)], axis=1)
    return -1, n_in


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data", default="data/bfcl_router")
    ap.add_argument("--levels", default="names,desc,schema")
    ap.add_argument("--label", default="trained-0.6B")
    args = ap.parse_args()

    mx.set_memory_limit(48 * 1024**3)
    model, tok = base.load_base(args.base)
    if args.adapter:
        cfg_path = Path(args.adapter).parent / "adapter_config.json"
        rank, nl = 16, 5
        if cfg_path.exists():
            c = json.load(open(cfg_path)); rank = int(c.get("rank", rank)); nl = int(c.get("num_layers", nl))
        base.install_dora(model, nl, rank)
        model.update(tree_unflatten(list(mx.load(args.adapter).items())))
        mx.eval(model.parameters())
    model.eval()

    n2s = name2schema()
    holdout = [json.loads(l) for l in open(Path(args.data) / "holdout.jsonl")]
    out = {}
    for level in args.levels.split(","):
        correct, toks = 0, []
        for inst in holdout:
            pred, n_in = predict_index(model, tok, inst, n2s, level)
            toks.append(n_in)
            correct += (pred == int(inst["gold_position"]))
            mx.clear_cache()
        n = len(holdout)
        med = sorted(toks)[n // 2]
        out[level] = {"selection_acc": 100 * correct / n, "median_prompt_tok": med, "n": n}
        print(f"  [{args.label}/{level}] acc={100*correct/n:.1f}%  median_ctx={med}tok")

    res = {"label": args.label, "base": args.base, "adapter": args.adapter, "levels": out}
    p = Path("eval/bfcl") / f"headctx_{args.label}.json"
    p.write_text(json.dumps(res, indent=2))
    print(f"=== wrote {p} ===")


if __name__ == "__main__":
    main()
