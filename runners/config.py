"""
Shared configuration validation for runner scripts.

Validates user-facing parameters before they reach the RL engine,
producing clear error messages that identify WHAT is wrong, WHERE,
and what valid values are expected.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path regardless of invocation cwd.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from data.linerlib_loader import LINERLIBLoader
from neural.config import ArchitectureConfig


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_POLICIES = ("encoder_only", "encoder_decoder")
DEFAULT_INSTANCE = "Baltic"
DEFAULT_POLICY = "encoder_only"
DEFAULT_SEED = 42
DEFAULT_HIDDEN_DIM = 32        # Small for smoke / pipeline runs
DEFAULT_GAT_LAYERS = 1
DEFAULT_TRANSFORMER_LAYERS = 1
DEFAULT_TRANSFORMER_HEADS = 2  # Must divide hidden_dim
DEFAULT_LSTM_LAYERS = 1
DEFAULT_MAX_UPDATES = 3        # Pipeline smoke: minimal
DEFAULT_CHECKPOINT_FREQUENCY = 100


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class RunnerConfigError(ValueError):
    """Raised when user-facing configuration is invalid."""

    def __init__(self, message: str, field: str = "", hint: str = "") -> None:
        self.field = field
        self.hint = hint
        parts = [message]
        if field:
            parts.append(f"Field: {field}")
        if hint:
            parts.append(f"Hint: {hint}")
        super().__init__("\n".join(parts))


# ---------------------------------------------------------------------------
# Instance validation
# ---------------------------------------------------------------------------

def validate_instance(instance_name: str) -> str:
    """Return the validated instance name (canonical form)."""
    loader = LINERLIBLoader("data")
    available = loader.available_instances()
    name_lower = instance_name.strip().lower()
    for avail in available:
        if avail.lower() == name_lower:
            return avail
    raise RunnerConfigError(
        f"Unknown instance '{instance_name}'.",
        field="instance",
        hint=f"Available instances: {', '.join(available)}",
    )


# ---------------------------------------------------------------------------
# Policy validation
# ---------------------------------------------------------------------------

def validate_policy(policy: str) -> str:
    """Return the validated policy type."""
    if policy is None:
        raise RunnerConfigError(
            f"'policy' must be a string, got None.",
            field="policy",
            hint=f"Valid values: {', '.join(VALID_POLICIES)}",
        )
    p = policy.strip().lower()
    if p not in VALID_POLICIES:
        raise RunnerConfigError(
            f"Invalid policy type '{policy}'.",
            field="policy",
            hint=f"Valid values: {', '.join(VALID_POLICIES)}",
        )
    return p


# ---------------------------------------------------------------------------
# Numeric validators
# ---------------------------------------------------------------------------

def _validate_positive_float(value: Any, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise RunnerConfigError(
            f"'{name}' must be a number, got {type(value).__name__}: {value!r}.",
            field=name,
        )
    import math
    if not math.isfinite(f) or f <= 0:
        raise RunnerConfigError(
            f"'{name}' must be a finite positive number, got {f}.",
            field=name,
            hint="Use a positive floating-point number.",
        )
    return f


def _validate_non_negative_float(value: Any, name: str) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise RunnerConfigError(
            f"'{name}' must be a number, got {type(value).__name__}: {value!r}.",
            field=name,
        )
    if f < 0:
        raise RunnerConfigError(
            f"'{name}' must be >= 0, got {f}.",
            field=name,
            hint="Use a non-negative floating-point number.",
        )
    return f


def _validate_int(value: Any, name: str, min_val: int = 1) -> int:
    try:
        i = int(value)
    except (TypeError, ValueError):
        raise RunnerConfigError(
            f"'{name}' must be an integer, got {type(value).__name__}: {value!r}.",
            field=name,
        )
    if i < min_val:
        raise RunnerConfigError(
            f"'{name}' must be >= {min_val}, got {i}.",
            field=name,
            hint=f"Use an integer >= {min_val}.",
        )
    return i


def _validate_in_range(value: Any, name: str, lo: float, hi: float) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise RunnerConfigError(
            f"'{name}' must be a number, got {type(value).__name__}: {value!r}.",
            field=name,
        )
    if not (lo <= f <= hi):
        raise RunnerConfigError(
            f"'{name}' must be in [{lo}, {hi}], got {f}.",
            field=name,
            hint=f"Use a value between {lo} and {hi}.",
        )
    return f


# ---------------------------------------------------------------------------
# Full configuration validation
# ---------------------------------------------------------------------------

def validate_runner_config(
    instance: str,
    policy: str,
    *,
    seed: Any = DEFAULT_SEED,
    max_updates: Any = DEFAULT_MAX_UPDATES,
    num_envs: Any = 1,
    steps_per_env: Any = 50,
    minibatch_size: Any = 32,
    learning_rate: Any = 2e-4,
    gamma: Any = 1.0,
    gae_lambda: Any = 0.9,
    ppo_epochs: Any = 2,
    clip_epsilon: Any = 0.2,
    target_kl: Any = 0.1,
    entropy_coefficient: Any = 0.05,
    value_coefficient: Any = 0.5,
    hidden_dim: Any = DEFAULT_HIDDEN_DIM,
    gat_layers: Any = DEFAULT_GAT_LAYERS,
    transformer_layers: Any = DEFAULT_TRANSFORMER_LAYERS,
    transformer_heads: Any = DEFAULT_TRANSFORMER_HEADS,
    lstm_layers: Any = DEFAULT_LSTM_LAYERS,
    checkpoint_frequency: Any = DEFAULT_CHECKPOINT_FREQUENCY,
) -> Dict[str, Any]:
    """
    Validate all runner-script parameters and return a clean dict.

    Raises RunnerConfigError on any invalid value.
    """
    # Instance
    inst = validate_instance(instance)

    # Policy
    pol = validate_policy(policy)

    # Numeric fields
    seed_val = _validate_int(seed, "seed", min_val=0)
    max_upd = _validate_int(max_updates, "max_updates", min_val=1)
    num_envs_val = _validate_int(num_envs, "num_envs", min_val=1)
    steps_val = _validate_int(steps_per_env, "steps_per_env", min_val=1)
    mb_size = _validate_int(minibatch_size, "minibatch_size", min_val=1)
    lr = _validate_positive_float(learning_rate, "learning_rate")
    gamma_val = _validate_in_range(gamma, "gamma", 0.0, 1.0)
    gae = _validate_in_range(gae_lambda, "gae_lambda", 0.0, 1.0)
    ppo_ep = _validate_int(ppo_epochs, "ppo_epochs", min_val=1)
    clip = _validate_in_range(clip_epsilon, "clip_epsilon", 0.0, 1.0)
    if clip <= 0:
        raise RunnerConfigError(
            "'clip_epsilon' must be > 0, got {clip}.",
            field="clip_epsilon",
        )
    target_kl_val = _validate_positive_float(target_kl, "target_kl")
    ent_coeff = _validate_non_negative_float(entropy_coefficient, "entropy_coefficient")
    val_coeff = _validate_non_negative_float(value_coefficient, "value_coefficient")
    hidden = _validate_int(hidden_dim, "hidden_dim", min_val=1)
    gat_l = _validate_int(gat_layers, "gat_layers", min_val=1)
    trans_l = _validate_int(transformer_layers, "transformer_layers", min_val=1)
    trans_h = _validate_int(transformer_heads, "transformer_heads", min_val=1)
    lstm_l = _validate_int(lstm_layers, "lstm_layers", min_val=1)
    ckpt_freq = _validate_int(checkpoint_frequency, "checkpoint_frequency", min_val=1)

    # Cross-field checks
    if hidden % trans_h != 0:
        raise RunnerConfigError(
            f"'hidden_dim' ({hidden}) must be divisible by "
            f"'transformer_heads' ({trans_h}).",
            field="hidden_dim",
            hint=f"Choose hidden_dim as a multiple of {trans_h}, "
                 f"e.g. {trans_h * max(1, hidden // trans_h)}.",
        )

    return {
        "instance": inst,
        "policy": pol,
        "seed": seed_val,
        "max_updates": max_upd,
        "num_envs": num_envs_val,
        "steps_per_env": steps_val,
        "minibatch_size": mb_size,
        "learning_rate": lr,
        "gamma": gamma_val,
        "gae_lambda": gae,
        "ppo_epochs": ppo_ep,
        "clip_epsilon": clip,
        "target_kl": target_kl_val,
        "entropy_coefficient": ent_coeff,
        "value_coefficient": val_coeff,
        "hidden_dim": hidden,
        "gat_layers": gat_l,
        "transformer_layers": trans_l,
        "transformer_heads": trans_h,
        "lstm_layers": lstm_l,
        "checkpoint_frequency": ckpt_freq,
    }


# ---------------------------------------------------------------------------
# Architecture config builder
# ---------------------------------------------------------------------------

def build_architecture_config(
    hidden_dim: int,
    gat_layers: int,
    transformer_layers: int,
    transformer_heads: int,
    lstm_layers: int,
) -> ArchitectureConfig:
    """Build and validate an ArchitectureConfig from runner parameters."""
    return ArchitectureConfig(
        hidden_dim=hidden_dim,
        gat_layers=gat_layers,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        lstm_layers=lstm_layers,
    )
