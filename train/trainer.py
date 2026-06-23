"""Head-parameterized SQRL trainer — the self-contained loop.

The heart of the port (docs/PORT_PLAN.md §4). A single train() drives ANY head
that satisfies the Head protocol (load_data / build_batch / loss / eval /
render_inference): value_and_grad over head.loss, AdamW, seed-sweep, eval-gated
early-stop, save best-on-holdout adapter. The router is the first consumer; the
the head module implements a small Head protocol; new heads plug in via HEADS.

A clean, self-contained training loop: it
hardcodes the router's eval (eval_holdout), two named holdout sets (slm/v29), a
template_sha8 lock, and ~40 argparse flags. Here the head supplies its own loss
and eval; the trainer owns only the orchestration. Reuses shared/base for the
frozen-base + DoRA install, so the recipe (TrainConfig) is the only knob surface.

CLI:  python -m train.trainer --head router --data data/bfcl_router \
        [--lambda 0.10 --star-temp 0.08 --seeds 0,1,2 --max-steps 600]
"""
from __future__ import annotations

import argparse
import importlib
import json
import time
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from shared import base
from train.config import TrainConfig

HEADS = {  # head name → module path implementing the Head protocol
    "router": "heads.router",  # the function-call selector head (this release)
}


def _install_memory_guards():
    """Cap MLX memory + bound the Metal buffer cache.

    HOST-STABILITY (load-bearing): MLX's Metal allocator grows an unbounded buffer
    CACHE across steps — on a large corpus (the 10k-instance Rea run) it ballooned
    to ~21GB while the actual working set stayed ~2.3GB, and combined with the eval
    pass it exhausted the kernel VM compressor and REBOOTED the host (2026-06-21).
    The training/eval loops call mx.clear_cache() each step; these limits are the
    belt-and-suspenders hard nets so a runaway alloc RAISES instead of panicking
    the machine. 24GB ceiling / 2GB cache is ample for a 0.6B + DoRA (active ~2.3GB,
    in-step peak ~10.5GB) and far below physical RAM.
    """
    mx.set_memory_limit(24 * 1024**3)   # hard ceiling — raise, don't reboot
    mx.set_cache_limit(2 * 1024**3)     # bound the buffer cache


def _save_adapter(model, path: Path):
    """Save only the trainable DoRA params (lora_a, lora_b, m)."""
    flat = tree_flatten(model.trainable_parameters())
    mx.save_safetensors(str(path), dict(flat))


def train_one_seed(head, train_insts, holdout, cfg: TrainConfig, seed: int,
                   out_dir: Path):
    """Train one seed; return (best_holdout_metric, best_adapter_path)."""
    _install_memory_guards()
    model, tok = base.load_base(cfg.base_model)
    n_params = base.install_dora(model, cfg.num_layers, cfg.rank)

    optimizer = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    loss_and_grad = nn.value_and_grad(model, lambda m, b: head.loss(m, b, cfg))

    rng = np.random.RandomState(seed)
    idx = np.arange(len(train_insts))
    seed_dir = out_dir / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(seed_dir / "train_log.jsonl", "w")

    best_metric = -1.0
    plateau = 0
    step = 0
    wall0 = time.time()
    model.train()

    print(f"[seed {seed}] base={cfg.base_model} dora_params={n_params:,} "
          f"recipe={cfg.recipe_tag()}", flush=True)

    done = False
    while not done:
        rng.shuffle(idx)
        for b0 in range(0, len(idx), cfg.batch_size):
            bidx = idx[b0:b0 + cfg.batch_size]
            insts = [train_insts[int(i)] for i in bidx]
            try:
                batch = head.build_batch(tok, insts)
            except Exception as e:                       # defensive: skip bad batch
                print(f"  batch error step {step}: {e}", flush=True)
                continue
            loss, grads = loss_and_grad(model, batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            mx.clear_cache()   # release the Metal buffer cache (see _install_memory_guards)
            step += 1

            if step % 50 == 0:
                log_f.write(json.dumps({"step": step, "loss": float(loss),
                                        "wall": round(time.time() - wall0, 1)}) + "\n")
                log_f.flush()
                print(f"  step {step:>5} loss={float(loss):.4f} "
                      f"wall={time.time()-wall0:.0f}s", flush=True)

            if step % cfg.eval_every == 0:
                model.eval()
                metric = head.eval(model, tok, holdout, max_n=cfg.eval_max_n)
                model.train()
                log_f.write(json.dumps({"step": step, "holdout": metric}) + "\n")
                log_f.flush()
                print(f"  >>> eval step {step}: holdout={metric:.3f} "
                      f"(best {max(best_metric,0):.3f})", flush=True)
                if metric > best_metric:
                    best_metric = metric
                    plateau = 0
                    _save_adapter(model, seed_dir / "adapters.safetensors")
                    (seed_dir / "adapter_config.json").write_text(json.dumps({
                        "head": head.NAME, "base_model": cfg.base_model,
                        "num_layers": cfg.num_layers, "rank": cfg.rank,
                        "star_lambda": cfg.star_lambda, "star_temp": cfg.star_temp,
                        "recipe": cfg.recipe_tag(), "seed": seed,
                        "best_holdout": metric, "step": step,
                    }, indent=2))
                else:
                    plateau += 1
                    if plateau >= cfg.early_stop_patience:
                        print(f"  early-stop seed {seed} at step {step} "
                              f"(plateau {plateau})", flush=True)
                        done = True
                        break
            if step >= cfg.max_steps:
                done = True
                break
    log_f.close()
    return best_metric, seed_dir / "adapters.safetensors"


def train(head, data_dir: str, cfg: TrainConfig):
    """Seed-sweep; return (best_metric, best_adapter_path) over cfg.seeds."""
    train_insts, holdout = head.load_data(data_dir)
    out_dir = Path(cfg.save_dir) if cfg.save_dir else \
        Path("adapters") / f"{head.NAME}-{cfg.recipe_tag()}"
    print(f"=== train head={head.NAME} n_train={len(train_insts)} "
          f"n_holdout={len(holdout)} seeds={cfg.seeds} → {out_dir} ===", flush=True)

    results = []
    for seed in cfg.seeds:
        metric, path = train_one_seed(head, train_insts, holdout, cfg, seed, out_dir)
        results.append((metric, seed, path))
        print(f"[seed {seed}] best holdout = {metric:.3f}", flush=True)

    best_metric, best_seed, best_path = max(results, key=lambda r: r[0])
    summary = {"head": head.NAME, "recipe": cfg.recipe_tag(),
               "per_seed": [{"seed": s, "holdout": m} for m, s, _ in results],
               "best_seed": best_seed, "best_holdout": best_metric,
               "best_adapter": str(best_path)}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"=== best seed {best_seed}: holdout={best_metric:.3f} → {best_path} ===",
          flush=True)
    return best_metric, best_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="router", choices=list(HEADS))
    ap.add_argument("--data", required=True, help="data dir for the head")
    ap.add_argument("--base", default=None, help="override base_model")
    ap.add_argument("--lambda", dest="star_lambda", type=float, default=None)
    ap.add_argument("--star-temp", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--eval-max-n", type=int, default=None)
    ap.add_argument("--rank", type=int, default=None, help="DoRA rank (scale for big bases)")
    ap.add_argument("--num-layers", type=int, default=None, help="DoRA last-N layers")
    ap.add_argument("--seeds", default=None, help="comma list, e.g. 0,1,2")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()

    head = importlib.import_module(HEADS[args.head])
    cfg = TrainConfig()
    over = {}
    if args.base is not None: over["base_model"] = args.base
    if args.star_lambda is not None: over["star_lambda"] = args.star_lambda
    if args.star_temp is not None: over["star_temp"] = args.star_temp
    if args.lr is not None: over["lr"] = args.lr
    if args.max_steps is not None: over["max_steps"] = args.max_steps
    if args.eval_every is not None: over["eval_every"] = args.eval_every
    if args.eval_max_n is not None: over["eval_max_n"] = args.eval_max_n
    if args.rank is not None: over["rank"] = args.rank
    if args.num_layers is not None: over["num_layers"] = args.num_layers
    if args.seeds is not None: over["seeds"] = tuple(int(s) for s in args.seeds.split(","))
    if args.save_dir is not None: over["save_dir"] = args.save_dir
    cfg = replace(cfg, **over)

    train(head, args.data, cfg)


if __name__ == "__main__":
    main()
