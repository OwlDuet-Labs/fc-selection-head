"""Context-richness sweep: does the frontier/trained-FC ranking vs our 0.6B head change
with how much tool documentation each model is given?

Motivation: our selector head selects from a list of bare function NAMES (train/batch.py
render: `[i] {id}` — no description, no schema). The trained-FC and frontier baselines, in
contrast, were given the FULL BFCL schema (parameters, types, descriptions — ~7x the tokens).
So the existing comparison already handicaps our head on context. This script measures every
model under matched context levels:

  level "names"  : `[i] name`                      (what the head uses; minimal context)
  level "desc"   : `[i] name: short description`   (BFCL's terse desc)
  level "schema" : full BFCL function schema JSON   (params+types; the high-context setting)

For each instance we log n_candidates and the prompt token count, so accuracy can be plotted
against actual context size. External/frontier models still emit a native call; we map the
chosen function NAME back to a candidate index and score selection (arguments not graded),
identical to trained_fc_baseline_eval.py.

Usage:
  # local trained-FC / vanilla through HF:
  python eval/bfcl/context_sweep_eval.py --backend hf --model Salesforce/xLAM-2-1b-fc-r \
      --label xlam-1b --levels names,desc,schema
  # frontier API:
  python eval/bfcl/context_sweep_eval.py --backend anthropic --model claude-sonnet-4-6 \
      --label sonnet --levels names,desc,schema
"""
import argparse
import json
import os
import re
from pathlib import Path

RAW_BFCL = Path("/Volumes/X10/owl/app/daw-mcp/u-factoring-bfcl/.venv/lib/python3.11/"
                "site-packages/bfcl_eval/data/BFCL_v4_live_multiple.json")


def build_name2schema():
    n = {}
    for line in open(RAW_BFCL):
        for f in json.loads(line)["function"]:
            n[f["name"]] = f
    return n


def candidate_block(candidates, name2schema, level):
    """Render the candidate list at a given context level."""
    lines = []
    for i, c in enumerate(candidates):
        tag = f"[{i+1}]"
        if level == "names":
            lines.append(f"  {tag} {c['id']}")
        elif level == "desc":
            lines.append(f"  {tag} {c['id']}: {c.get('desc','')}")
        else:  # schema
            sch = name2schema.get(c["id"], {"name": c["id"]})
            lines.append(f"  {tag} {json.dumps(sch)}")
    return "\n".join(lines)


SYS = ("You select the single most appropriate function for the user request from the "
       "numbered candidate list. Reply with ONLY the function name of your choice.")


def make_prompt(inst, name2schema, level):
    block = candidate_block(inst["candidates"], name2schema, level)
    return (f"{SYS}\n\nRequest: {inst['prompt']}\n\nCandidates:\n{block}\n\n"
            f"Answer with the chosen function name:")


_NAME_PATS = [
    re.compile(r'"tool_calls"\s*:\s*\[\s*\{\s*"name"\s*:\s*"([^"]+)"'),
    re.compile(r'"name"\s*:\s*"([^"]+)"'),
    re.compile(r'\b([A-Za-z_][A-Za-z0-9_.]*)\s*\('),
]


def parse_name(text, valid):
    cands = []
    for v in valid:                 # exact substring match first (most robust here)
        if v in text:
            cands.append((text.index(v), v))
    if cands:
        return min(cands)[1]        # earliest-mentioned valid name
    for pat in _NAME_PATS:
        for m in pat.findall(text):
            for v in valid:
                if v.endswith(m) or v.split(".")[-1] == m:
                    return v
    return None


def make_backend(backend, model):
    if backend == "anthropic":
        import anthropic
        cl = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_ITERATE_BENCH_API_KEY")
                                 or os.environ["ANTHROPIC_API_KEY"], timeout=30, max_retries=3)
        def ask(p):
            m = cl.messages.create(model=model, max_tokens=32, temperature=0,
                                   messages=[{"role": "user", "content": p}])
            return "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
        def ntok(p):
            return len(p) // 4
        return ask, ntok
    if backend == "openai-chat":
        from openai import OpenAI
        cl = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        def ask(p):
            kw = {"model": model, "messages": [{"role": "user", "content": p}]}
            if model.startswith("gpt-5"):
                kw["max_completion_tokens"] = 32
            else:
                kw["max_tokens"] = 32; kw["temperature"] = 0
            return cl.chat.completions.create(**kw).choices[0].message.content or ""
        return ask, (lambda p: len(p) // 4)
    # hf (local MLX) — trained-FC / vanilla models. Apply the model's OWN chat template so
    # xLAM/Hammer see their trained format (a raw string is off-distribution and unfair).
    from mlx_lm import load, generate
    model_obj, tok = load(model)
    def _wrap(p):
        try:
            return tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        except Exception:
            return p
    def ask(p):
        return generate(model_obj, tok, prompt=_wrap(p), max_tokens=32, verbose=False)
    def ntok(p):
        return len(tok.encode(_wrap(p)))
    return ask, ntok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["hf", "anthropic", "openai-chat"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--levels", default="names,desc,schema")
    ap.add_argument("--data", default="data/bfcl_router")
    ap.add_argument("--max-n", type=int, default=134)
    args = ap.parse_args()

    name2schema = build_name2schema()
    ask, ntok = make_backend(args.backend, args.model)
    holdout = [json.loads(l) for l in open(Path(args.data) / "holdout.jsonl")][: args.max_n]
    levels = args.levels.split(",")

    out_levels = {}
    for level in levels:
        correct, parsed, toks = 0, 0, []
        log = []
        for inst in holdout:
            valid = {c["id"] for c in inst["candidates"]}
            p = make_prompt(inst, name2schema, level)
            toks.append(ntok(p))
            txt = ask(p)
            name = parse_name(txt, valid)
            pick = next((j + 1 for j, c in enumerate(inst["candidates"]) if c["id"] == name), -1)
            ok = pick == int(inst["gold_position"])
            correct += ok; parsed += name is not None
            log.append({"gold": inst["gold"], "pick": name, "ok": ok, "n_cand": len(valid)})
        n = len(holdout)
        med = sorted(toks)[n // 2]
        out_levels[level] = {"selection_acc": 100 * correct / n, "parse_rate": 100 * parsed / n,
                             "median_prompt_tok": med, "n": n, "log": log}
        print(f"  [{args.label}/{level}] acc={100*correct/n:.1f}%  parse={100*parsed/n:.0f}%  "
              f"median_ctx={med}tok")

    res = {"label": args.label, "backend": args.backend, "model": args.model, "levels": out_levels}
    outp = Path("eval/bfcl") / f"ctxsweep_{args.label}.json"
    outp.write_text(json.dumps(res, indent=2))
    print(f"=== {args.label}: wrote {outp} ===")


if __name__ == "__main__":
    main()
