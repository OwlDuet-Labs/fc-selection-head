"""Apples-to-apples selection eval for EXTERNAL trained function-calling models.

The paper's selector head emits {"index": N} into a candidate list. Published trained
FC models (xLAM, Hammer) instead emit a native function call
{"name": "...", "arguments": {...}}. To compare fairly on the SAME selection task and the
SAME holdout, we:

  1. Rehydrate the FULL BFCL function schema for every candidate (our split stores only
     id+desc; xLAM/Hammer need the parameter schema to call). Names -> raw
     BFCL_v4_live_multiple.json `function` entries (100% coverage, verified).
  2. Render a native FC prompt: the user query + the candidate functions as tools.
  3. Run the model, parse the emitted function NAME (handles {"name":...}, [{"name":...}],
     <tool_call>...</tool_call>, ```json fences, and bare name(...) forms).
  4. Map the chosen name back to its candidate index and score against gold_position.

This scores "did the model pick the right function from this candidate set" — exactly our
selection metric — regardless of the model's output format. Selection-only: we DO NOT grade
the emitted arguments (that would be a different, harder metric than our head's task).

Usage:
  python eval/bfcl/trained_fc_baseline_eval.py --model mlx-community/xLAM-2-1b-fc-r --label xlam-1b
  python eval/bfcl/trained_fc_baseline_eval.py --model mlx-community/Hammer2.1-1.5b   --label hammer-1.5b
"""
import argparse
import json
import re
from pathlib import Path

from mlx_lm import load, generate

RAW_BFCL = Path("/Volumes/X10/owl/app/daw-mcp/u-factoring-bfcl/.venv/lib/python3.11/"
                "site-packages/bfcl_eval/data/BFCL_v4_live_multiple.json")


def build_name2schema():
    name2 = {}
    for line in open(RAW_BFCL):
        for f in json.loads(line)["function"]:
            name2[f["name"]] = f
    return name2


XLAM_FC_TASK = (
    "You are an expert in composing functions. You are given a question and a set of "
    "possible functions. Based on the question, you will need to make one function call "
    "to achieve the purpose. If none of the functions can be used, point it out. You "
    "should only return the function call in the tools-call format.")
XLAM_FC_FORMAT = (
    'The output MUST strictly adhere to the following JSON format:\n'
    '{"tool_calls": [{"name": "func_name", "arguments": {...}}]}')


def _xlam_fc_prompt(tokenizer, inst, schemas):
    """xLAM-*-fc-r custom format: its chat template ignores tools=, so we inject the
    candidate schemas into the instruction body the way the model was trained on."""
    tools_json = json.dumps(schemas, indent=2)
    instr = (f"{XLAM_FC_TASK}\n\n{XLAM_FC_FORMAT}\n\n"
             f"Available tools:\n{tools_json}\n\nQuestion: {inst['prompt']}")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instr}], tokenize=False, add_generation_prompt=True)


def render_prompt(tokenizer, inst, name2schema, model_id="", context_level="schema"):
    """Native function-call prompt: user query + candidate functions as tools.

    context_level controls how much per-candidate documentation goes in the tool spec, so the
    SAME validated native-format harness measures the context-richness axis:
      schema (default) : full BFCL schema (params + types + description)
      desc             : name + description only (no parameters)
      names            : name only (minimal context, matched to the selector head)"""
    schemas = []
    for c in inst["candidates"]:
        full = name2schema.get(c["id"]) or {"name": c["id"], "description": c.get("desc", ""),
                                             "parameters": {"type": "object", "properties": {}}}
        if context_level == "names":
            sch = {"name": full["name"], "parameters": {"type": "object", "properties": {}}}
        elif context_level == "desc":
            sch = {"name": full["name"], "description": full.get("description", c.get("desc", "")),
                   "parameters": {"type": "object", "properties": {}}}
        else:
            sch = full
        schemas.append(sch)
    ct = (tokenizer.chat_template or "")
    # xLAM-fc-r's template does NOT honor tools= (no 'tools' token) — inject manually.
    if "fc-r" in model_id.lower() or ("tools" not in ct.lower() and "xlam" in model_id.lower()):
        return _xlam_fc_prompt(tokenizer, inst, schemas)
    msgs = [{"role": "user", "content": inst["prompt"]}]
    try:
        p = tokenizer.apply_chat_template(
            msgs, tools=schemas, tokenize=False, add_generation_prompt=True)
        # guard: if the template silently dropped the tools, fall back to manual injection
        if schemas and schemas[0]["name"] not in p:
            return _xlam_fc_prompt(tokenizer, inst, schemas)
        return p
    except Exception:
        return _xlam_fc_prompt(tokenizer, inst, schemas)


_NAME_PATS = [
    re.compile(r'"tool_calls"\s*:\s*\[\s*\{\s*"name"\s*:\s*"([^"]+)"'),  # xLAM {"tool_calls":[{"name":..}]}
    re.compile(r'"name"\s*:\s*"([^"]+)"'),                 # {"name": "foo"}
    re.compile(r"'name'\s*:\s*'([^']+)'"),
    re.compile(r'<tool_call>\s*\{?\s*"?name"?\s*:?\s*"?([A-Za-z0-9_.]+)'),
    re.compile(r'```(?:json)?\s*\[?\s*\{\s*"name"\s*:\s*"([^"]+)"'),
    re.compile(r'\b([A-Za-z_][A-Za-z0-9_.]*)\s*\('),       # bare foo(...)
]


def parse_name(text, valid_names):
    """Extract the chosen function name; prefer a name that's in the candidate set."""
    cands = []
    for pat in _NAME_PATS:
        cands += pat.findall(text)
    # exact match to a candidate first
    for c in cands:
        if c in valid_names:
            return c
    # suffix/last-component match (e.g. predicted 'change_drink' vs id 'ChaDri.change_drink')
    for c in cands:
        for v in valid_names:
            if v.endswith(c) or v.split(".")[-1] == c or c.split(".")[-1] == v.split(".")[-1]:
                return v
    return cands[0] if cands else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--data", default="data/bfcl_router")
    ap.add_argument("--max-n", type=int, default=200)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--context-level", default="schema", choices=["names", "desc", "schema"])
    args = ap.parse_args()

    name2schema = build_name2schema()
    model, tokenizer = load(args.model)
    holdout = [json.loads(l) for l in open(Path(args.data) / "holdout.jsonl")][: args.max_n]

    log, correct, parsed = [], 0, 0
    for i, inst in enumerate(holdout):
        valid = {c["id"] for c in inst["candidates"]}
        prompt = render_prompt(tokenizer, inst, name2schema, model_id=args.model,
                               context_level=args.context_level)
        out = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        name = parse_name(out, valid)
        pick = next((j + 1 for j, c in enumerate(inst["candidates"]) if c["id"] == name), -1)
        ok = (pick == int(inst["gold_position"]))
        correct += ok
        parsed += (name is not None)
        log.append({"i": i, "gold": inst["gold"], "gold_pos": inst["gold_position"],
                    "pick_name": name, "pick_pos": pick, "correct": ok})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(holdout)}  acc={100*correct/(i+1):.1f}%  parsed={100*parsed/(i+1):.0f}%")

    n = len(holdout)
    res = {"label": args.label, "model": args.model, "data": args.data, "n": n,
           "selection_acc": 100 * correct / n, "parse_rate": 100 * parsed / n, "log": log}
    suffix = "" if args.context_level == "schema" else f"_{args.context_level}"
    out = Path("eval/bfcl") / f"trained_fc_{args.label}{suffix}.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"=== {args.label} ({args.model}): selection-acc {res['selection_acc']:.1f}% "
          f"(parse {res['parse_rate']:.0f}%, n={n}) → {out} ===")


if __name__ == "__main__":
    main()
