"""On-device latency as a function of INPUT CONTEXT LENGTH (run on a real M1).

The point: feeding a model more tool documentation (full schemas vs bare names) raises accuracy
for capable models, but the extra context costs prefill latency. This measures that cost on
device for the small bases the selector head runs on, so the accuracy-vs-context story can be
paired with a latency-vs-context story (the on-device viability argument).

Holds the generation short (the selector emits a 1-2 token index) and varies the INPUT length:
context_tokens in {100, 250, 500, 860, 1500, 2500}. 860 ~ the median full-schema BFCL prompt;
100 ~ the names-only prompt the trained head actually uses.

Hardware-honest: prints the detected chip + RAM into the output JSON. Run this ON THE TARGET
DEVICE (a real Apple M1) so the numbers are genuine — do NOT run on a different Mac and label
it M1.

STANDALONE: the only dependencies are `mlx` and `mlx-lm` (pip install mlx mlx-lm). No
repo-internal imports, no dataset files needed (context is synthetic filler tokens), no
hardcoded paths. Drop this single file anywhere on the M1 and run it. Models download from
HuggingFace (mlx-community/Qwen3-*) on first use.

Usage (on the M1, from anywhere):
  python latency_vs_context.py --models 0.6B,1.7B --quants q4,q6 \
      --contexts 100,250,500,860,1500 --reps 5
Writes latency_vs_context_<chip>.json next to the script and prints the JSON to stdout.
"""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

REPO = {  # explicit per-(size,quant) MLX repos so q4/q6/bf16 are genuine, not on-the-fly
    ("0.6B", "q4"): "mlx-community/Qwen3-0.6B-4bit",
    ("0.6B", "q6"): "mlx-community/Qwen3-0.6B-6bit",
    ("0.6B", "bf16"): "mlx-community/Qwen3-0.6B-bf16",
    ("1.7B", "q4"): "mlx-community/Qwen3-1.7B-4bit",
    ("1.7B", "q6"): "mlx-community/Qwen3-1.7B-6bit",
    ("1.7B", "bf16"): "mlx-community/Qwen3-1.7B-bf16",
    ("4B", "q4"): "mlx-community/Qwen3-4B-4bit",
    ("4B", "q6"): "mlx-community/Qwen3-4B-6bit",
    ("4B", "bf16"): "mlx-community/Qwen3-4B-bf16",
}


def chip_info():
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        chip = platform.processor() or "unknown"
    try:
        ram = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode()) // 1024**3
    except Exception:
        ram = 0
    return chip, ram


def make_input(tok, n_ctx):
    """A prompt of approximately n_ctx tokens (filler that looks like a candidate list)."""
    filler = ("[%d] some.function_name: a description of what this tool does and its "
              "parameters, types, and usage. ")
    s = "Select the correct function for the request.\n"
    i = 0
    while len(tok.encode(s)) < n_ctx:
        s += filler % i
        i += 1
    ids = tok.encode(s)[:n_ctx]
    return mx.array(ids, dtype=mx.int32)[None, :]


def time_one(model, ids, gen_tokens=2):
    """Prefill + short greedy decode; return wall ms and peak GB."""
    mx.eval(model.parameters())
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    cur = ids
    logit = model(cur)[:, -1, :]
    nxt = mx.argmax(logit, axis=-1)
    mx.eval(nxt)
    for _ in range(gen_tokens - 1):
        cur = mx.concatenate([cur, nxt[None, :]], axis=1)
        logit = model(cur)[:, -1, :]
        nxt = mx.argmax(logit, axis=-1)
        mx.eval(nxt)
    ms = (time.perf_counter() - t0) * 1000
    peak_gb = mx.get_peak_memory() / 1024**3
    return ms, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.6B,1.7B")
    ap.add_argument("--quants", default="q4,q6")
    ap.add_argument("--contexts", default="100,250,500,860,1500,2500")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="", help="output JSON path (default: next to this script)")
    args = ap.parse_args()

    chip, ram = chip_info()
    print(f"device: {chip}, {ram} GB")
    contexts = [int(x) for x in args.contexts.split(",")]
    runs = []
    for msize in args.models.split(","):
        for q in args.quants.split(","):
            repo = REPO.get((msize, q))
            if repo is None:
                print(f"  (skip {msize}/{q}: no repo mapped)")
                continue
            model, tok = load(repo)
            for n_ctx in contexts:
                ids = make_input(tok, n_ctx)
                # warmup
                time_one(model, ids)
                samples = [time_one(model, ids) for _ in range(args.reps)]
                ms = sorted(s[0] for s in samples)[len(samples) // 2]
                peak = max(s[1] for s in samples)
                runs.append({"model": msize, "quant": q, "context_tokens": n_ctx,
                             "median_ms": round(ms, 1), "peak_gb": round(peak, 2)})
                print(f"  {msize}/{q}  ctx={n_ctx:5}tok  median={ms:7.1f}ms  peak={peak:.2f}GB")
            del model
            mx.clear_cache()

    out = {"chip": chip, "ram_gb": ram, "reps": args.reps, "runs": runs}
    tag = chip.split()[1].lower() if len(chip.split()) > 1 else "dev"
    # Write next to this script (portable: works no matter the cwd), unless --out given.
    p = Path(args.out) if args.out else Path(__file__).resolve().parent / f"latency_vs_context_{tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"=== wrote {p} (device: {chip}, {ram}GB) ===")
    print(json.dumps(out, indent=2))  # also dump to stdout so you can paste it back


if __name__ == "__main__":
    main()
