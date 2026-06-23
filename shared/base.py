"""Shared frozen base + DoRA installer for SQRL heads.

Base-model loading + DoRA install for the frozen-base selector head.
(`load` + `install_dora`). All four heads adapt the SAME frozen Qwen3-0.6B base
with DoRA on the last N attention blocks; this module is the single place that
loads the base and installs the adapter, parameterized by `base_model` so a
ceiling probe can swap in a larger base (1.7B/4B/...) without forking the head
trainers.

The verified hook: `model.model(inputs)` returns the pre-lm_head hidden state
(used by the STAR contrastive path); `model(inputs)` returns logits. Both heads
rely on that distinction — do not change it.
"""
from __future__ import annotations

DEFAULT_BASE = "mlx-community/Qwen3-0.6B-bf16"

# DoRA target keys — the attention projections, identical across heads.
_DORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]


def load_base(base_model: str = DEFAULT_BASE):
    """Load (model, tokenizer) for the frozen base via mlx_lm. No adapter yet."""
    from mlx_lm import load
    return load(base_model)


def install_dora(model, num_layers: int = 5, rank: int = 16,
                 scale: float = 2.0, dropout: float = 0.05) -> int:
    """Apply DoRA to the last `num_layers` attention blocks; freeze the base.

    Returns the trainable-param count. The base stays frozen; only lora_a /
    lora_b / m (the DoRA magnitude) are trainable. For the 0.6B at
    num_layers=5/rank=16 this is ~844k (under the 1M router cap); a larger base
    has a bigger hidden size so the same config yields more params — raise the
    caller's param-cap accordingly (see the 1.7B probe: ~1.18M).
    """
    from mlx_lm.tuner.dora import DoRALinear
    from mlx_lm.tuner.utils import linear_to_lora_layers

    config = {"rank": rank, "scale": scale, "dropout": dropout, "keys": _DORA_KEYS}
    linear_to_lora_layers(model, num_layers=num_layers, config=config,
                          use_dora=True)
    model.freeze()
    trainable_n = 0

    def _unfreeze_dora(_, m):
        nonlocal trainable_n
        if isinstance(m, DoRALinear):
            m.unfreeze(keys=["lora_a", "lora_b", "m"], recurse=False)
            trainable_n += int(m.lora_a.size) + int(m.lora_b.size) + int(m.m.size)

    model.apply_to_modules(_unfreeze_dora)
    return trainable_n


def hidden_states(model, inputs):
    """Pre-lm_head hidden state [B, T, H] — the verified STAR-path hook."""
    return model.model(inputs)


def logits(model, inputs):
    """Vocabulary logits [B, T, V] — the standard CE-path forward."""
    return model(inputs)
