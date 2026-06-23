"""TrainConfig — the recipe dataclass for the self-contained SQRL trainer.

One immutable record carrying every knob the head-parameterized trainer needs:
base model, DoRA shape, optimization, and the STAR teacher-KD pair (lambda,
star_temp). Heads read it; the trainer reads it; nothing else holds these
defaults. The shipped router recipe is the default here so a no-arg
`TrainConfig()` reproduces the methods-paper STAR point.

A minimal training config. The original research trainer threaded ~40
flags (typed-context, string-target, shuffle-augmentation, confusion-map paths,
experiment-specific knobs; this keeps only
the knobs that define a recipe; experiment toggles become explicit fields if and
when a head needs them.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrainConfig:
    # --- base + adapter shape ---
    base_model: str = "Qwen/Qwen3-0.6B"
    num_layers: int = 5          # last-N attention layers get DoRA (shared/base default)
    rank: int = 16               # DoRA rank (GROW-CAPACITY memorized at higher; keep 16)

    # --- optimization ---
    lr: float = 1e-4
    max_steps: int = 600
    batch_size: int = 8
    eval_every: int = 50
    warmup_steps: int = 20
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    early_stop_patience: int = 4  # eval windows with no holdout improvement

    # --- STAR teacher-KD (the methods-paper winning pair) ---
    # lambda == 0 ⇒ the trainer adds ZERO star term (bit-identical to stock CE);
    # shared/star_loss enforces this invariant. 0.10/0.08 is the router winner.
    star_lambda: float = 0.10
    star_temp: float = 0.08

    # --- reproducibility ---
    seeds: tuple[int, ...] = (0, 1, 2)
    save_dir: str = ""           # empty ⇒ trainer derives from head name + recipe

    # --- eval ---
    eval_max_n: int = 200        # cap holdout rows scored per eval window

    def recipe_tag(self) -> str:
        """Short, filesystem-safe tag encoding the recipe (for adapter dirs/logs)."""
        return (f"L{int(self.star_lambda * 100):03d}"
                f"-T{int(self.star_temp * 1000):03d}"
                f"-r{self.rank}-n{self.num_layers}")
