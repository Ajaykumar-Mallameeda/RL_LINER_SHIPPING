"""
train_rl.py — Manual RL Training Control Interface for Liner Shipping Network Design.

This is the AUTHORITATIVE manual training entry point. It gives the researcher
full control over every experimental parameter while protecting scientific
invariants from accidental modification.

Usage:
    python train_rl.py                          # Run with PRESET config
    python train_rl.py --resume path/to/ckpt.pt  # Resume from checkpoint
    python train_rl.py --experiment MY_EXP       # Name your experiment

The script prints the ACTIVE TRAINING CONFIGURATION at startup so you always
know exactly what is running.

Scientific workflow this enforces:
    OBSERVE -> HYPOTHESIZE -> CONTROL ONE VARIABLE -> TRAIN -> MEASURE -> COMPARE -> DECIDE
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is on path.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runners.config import (
    DEFAULT_CHECKPOINT_FREQUENCY,
    DEFAULT_GAT_LAYERS,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_LSTM_LAYERS,
    DEFAULT_MAX_UPDATES,
    DEFAULT_POLICY,
    DEFAULT_SEED,
    DEFAULT_TRANSFORMER_HEADS,
    DEFAULT_TRANSFORMER_LAYERS,
    RunnerConfigError,
    validate_instance,
    validate_policy,
    validate_runner_config,
)
from runners.report import (
    VALIDATION_PASSED,
    VALIDATION_NA,
    collect_data_hashes,
    collect_reproducibility_info,
    validate_result,
    write_json,
    _border,
    _section,
)


# ======================================================================
# SECTION A: SCIENTIFIC INVARIANTS — DO NOT MODIFY WITHOUT EXPLICIT
#           REASON AND DOCUMENTATION
# ======================================================================
#
# These define the scientific problem. Changing them changes what you are
# studying. The paper's formulation is fixed here; to deviate, make an
# explicit code-level change and document it.
#
# [PAPER] Eq. 1   : eta = R_total − C_reject − C_handling − C_NDP
# [PAPER] Eq. 36  : R_{t+1} = (eta_{t+1} − eta_t) / eta_1   (normalized reward)
# [PAPER] Alg. 2  : Termination = vessel exhaustion OR demand satisfaction
# [PAPER] Sec 4.2 : Encoder-only: deterministic largest-available vessel
# [PAPER] Sec 4.3 : Encoder-decoder: learned stochastic vessel selection
# [PAPER] App. B  : MCF greedy Dijkstra on expanded graph
# [PAPER] Tab 5   : Paper architecture frozen values below
# ======================================================================

# --- Objective & Reward Invariants ---
# eta is minimized (more negative = worse); reward is incremental normalized eta.
# These equations are hard-coded in env/environment.py and mcf/costs.py.
# Do not modify them here — they are structural invariants of the engine.

# --- Fleet Semantics (Soft, per paper relaxation philosophy) ---
# Draft constraints are soft: C_unused handles economically.
# See FINAL_SCIENTIFIC_VERIFICATION_REPORT.md §5 for full justification.

# --- Paper Architecture (Table 5) ---
# These are the paper's frozen architectural constants.
# To run non-paper dimensions, change PRESET or override explicitly below.
PAPER_HIDDEN_DIM: int = 512
PAPER_GAT_LAYERS: int = 3
PAPER_TRANSFORMER_LAYERS: int = 3
PAPER_TRANSFORMER_HEADS: int = 8
PAPER_LSTM_LAYERS: int = 1

# --- Paper PPO Constants (Table 5) ---
PAPER_LEARNING_RATE: float = 2e-4
PAPER_GAMMA: float = 1.0
PAPER_GAE_LAMBDA: float = 0.9
PAPER_PPO_EPOCHS: int = 10
PAPER_CLIP_EPSILON: float = 0.2
PAPER_TARGET_KL: float = 0.1
PAPER_ENTROPY_COEFFICIENT: float = 0.05
PAPER_VALUE_COEFFICIENT: float = 0.5

# --- Transit Time ---
# Paper explicitly excludes transit time considerations (Section 1).
PAPER_TRANSIT_TIME_DISABLED: bool = True


# ======================================================================
# SECTION B: EXPERIMENTAL / TRAINING PARAMETERS
#             Edit these freely for your experiments.
# ======================================================================

# --- Preset Selection ---
# SMOKE  : Tiny model, few updates — sanity check only
# DEBUG  : Paper architecture, minimal envs/steps — diagnostic mode
# MANUAL : Your controlled parameters below (no automatic assumptions)
# PAPER  : Paper architecture + paper-compatible PPO settings
PRESET: str = "SMOKE"

# --- Instance ---
INSTANCE: str = "Baltic"
POLICY: str = "encoder_decoder"
SEED: int = 42

# --- Architecture ---
HIDDEN_DIM: int = 64          # Hidden embedding dimension
GAT_LAYERS: int = 2           # GAT message-passing layers
TRANSFORMER_LAYERS: int = 2   # Transformer encoder layers
TRANSFORMER_HEADS: int = 4    # Multi-head attention heads (must divide HIDDEN_DIM)
LSTM_LAYERS: int = 1          # LSTM decoder layers (encoder-decoder policy)

# --- PPO Training ---
LEARNING_RATE: float = 2e-4
GAMMA: float = 1.0
GAE_LAMBDA: float = 0.9
PPO_EPOCHS: int = 2          # PPO epochs per trajectory
MINIBATCH_SIZE: int = 32     # Minibatch size (independent of steps_per_env)
CLIP_EPSILON: float = 0.2
TARGET_KL: float = 0.1
ENTROPY_COEFFICIENT: float = 0.05   # Exploration knob — change one value at a time
VALUE_COEFFICIENT: float = 0.5

# --- Environment ---
NUM_ENVS: int = 1            # Parallel environments (paper uses 8–16)
STEPS_PER_ENV: int = 50      # Steps per environment per rollout
MAX_UPDATES: int = 10        # Total PPO update iterations
CHECKPOINT_FREQUENCY: int = 5  # Save checkpoint every N updates

# --- Perturbation (Paper Section 6.1) ---
PERTURBATION_FRACTION: float = 0.0    # 0.0 = no perturbation
N_PERTURBED_INSTANCES: int = 0        # Number of perturbed instances to generate

# --- Transit Time ---
APPLY_TRANSITTIME_REVISION: bool = False  # Paper excludes transit time

# --- Resume Control ---
RESUME_FROM_CHECKPOINT: str = ""  # Path to .pt checkpoint to resume from (empty = start fresh)


# ======================================================================
# END MANUAL TRAINING CONFIGURATION
# ======================================================================


# ======================================================================
# PRESETS — Named experiment configurations
# ======================================================================

_PRESETS: Dict[str, Dict[str, Any]] = {
    "SMOKE": {
        "description": "Tiny model, very few updates — sanity check only",
        "instance": "Baltic",
        "policy": "encoder_only",
        "hidden_dim": 16,
        "gat_layers": 1,
        "transformer_layers": 1,
        "transformer_heads": 2,
        "lstm_layers": 1,
        "learning_rate": 2e-4,
        "gamma": 1.0,
        "gae_lambda": 0.9,
        "ppo_epochs": 1,
        "minibatch_size": 16,
        "clip_epsilon": 0.2,
        "target_kl": 10.0,   # Disable KL LR adaptation for smoke
        "entropy_coefficient": 0.05,
        "value_coefficient": 0.5,
        "num_envs": 1,
        "steps_per_env": 20,
        "max_updates": 3,
        "checkpoint_frequency": 100,
        "perturbation_fraction": 0.0,
        "n_perturbed_instances": 0,
        "apply_transittime_revision": False,
        "seed": 42,
    },
    "DEBUG": {
        "description": "Paper architecture, tiny envs/steps — diagnostic mode",
        "instance": "Baltic",
        "policy": "encoder_decoder",
        "hidden_dim": PAPER_HIDDEN_DIM,
        "gat_layers": PAPER_GAT_LAYERS,
        "transformer_layers": PAPER_TRANSFORMER_LAYERS,
        "transformer_heads": PAPER_TRANSFORMER_HEADS,
        "lstm_layers": PAPER_LSTM_LAYERS,
        "learning_rate": PAPER_LEARNING_RATE,
        "gamma": PAPER_GAMMA,
        "gae_lambda": PAPER_GAE_LAMBDA,
        "ppo_epochs": PAPER_PPO_EPOCHS,
        "minibatch_size": 32,
        "clip_epsilon": PAPER_CLIP_EPSILON,
        "target_kl": PAPER_TARGET_KL,
        "entropy_coefficient": PAPER_ENTROPY_COEFFICIENT,
        "value_coefficient": PAPER_VALUE_COEFFICIENT,
        "num_envs": 1,
        "steps_per_env": 20,
        "max_updates": 5,
        "checkpoint_frequency": 100,
        "perturbation_fraction": 0.0,
        "n_perturbed_instances": 0,
        "apply_transittime_revision": False,
        "seed": 42,
    },
    "MANUAL": {
        "description": "User-controlled parameters, no automatic assumptions",
        # Values filled from SECTION B above at runtime
    },
    "PAPER": {
        "description": "Paper architecture + paper-compatible PPO + transit-time disabled",
        "instance": "Baltic",
        "policy": "encoder_decoder",
        "hidden_dim": PAPER_HIDDEN_DIM,
        "gat_layers": PAPER_GAT_LAYERS,
        "transformer_layers": PAPER_TRANSFORMER_LAYERS,
        "transformer_heads": PAPER_TRANSFORMER_HEADS,
        "lstm_layers": PAPER_LSTM_LAYERS,
        "learning_rate": PAPER_LEARNING_RATE,
        "gamma": PAPER_GAMMA,
        "gae_lambda": PAPER_GAE_LAMBDA,
        "ppo_epochs": PAPER_PPO_EPOCHS,
        "minibatch_size": 64,
        "clip_epsilon": PAPER_CLIP_EPSILON,
        "target_kl": PAPER_TARGET_KL,
        "entropy_coefficient": PAPER_ENTROPY_COEFFICIENT,
        "value_coefficient": PAPER_VALUE_COEFFICIENT,
        "num_envs": 8,
        "steps_per_env": 100,
        "max_updates": 200,
        "checkpoint_frequency": 50,
        "perturbation_fraction": 0.10,
        "n_perturbed_instances": 100,
        "apply_transittime_revision": False,
        "seed": 42,
    },
}


def _apply_preset(preset_name: str) -> None:
    """Override SECTION B values with preset defaults."""
    import sys as _sys
    preset = _PRESETS.get(preset_name.upper())
    if preset is None:
        raise ValueError(
            f"Unknown preset '{preset_name}'. "
            f"Valid presets: {', '.join(_PRESETS.keys())}"
        )
    if preset_name.upper() == "MANUAL":
        return  # MANUAL uses SECTION B as-is

    # Map snake_case preset keys -> PascalCase module variables
    KEY_MAP = {
        "instance": "INSTANCE",
        "policy": "POLICY",
        "hidden_dim": "HIDDEN_DIM",
        "gat_layers": "GAT_LAYERS",
        "transformer_layers": "TRANSFORMER_LAYERS",
        "transformer_heads": "TRANSFORMER_HEADS",
        "lstm_layers": "LSTM_LAYERS",
        "learning_rate": "LEARNING_RATE",
        "gamma": "GAMMA",
        "gae_lambda": "GAE_LAMBDA",
        "ppo_epochs": "PPO_EPOCHS",
        "minibatch_size": "MINIBATCH_SIZE",
        "clip_epsilon": "CLIP_EPSILON",
        "target_kl": "TARGET_KL",
        "entropy_coefficient": "ENTROPY_COEFFICIENT",
        "value_coefficient": "VALUE_COEFFICIENT",
        "num_envs": "NUM_ENVS",
        "steps_per_env": "STEPS_PER_ENV",
        "max_updates": "MAX_UPDATES",
        "checkpoint_frequency": "CHECKPOINT_FREQUENCY",
        "perturbation_fraction": "PERTURBATION_FRACTION",
        "n_perturbed_instances": "N_PERTURBED_INSTANCES",
        "apply_transittime_revision": "APPLY_TRANSITTIME_REVISION",
        "seed": "SEED",
    }

    mod = _sys.modules[__name__]
    for key, value in preset.items():
        if key == "description":
            continue
        upper_key = KEY_MAP.get(key)
        if upper_key is None:
            continue
        setattr(mod, upper_key, value)


def _get_mode_description(preset_name: str) -> str:
    preset = _PRESETS.get(preset_name.upper(), {})
    return preset.get("description", "Manual")


# ======================================================================
# CLI OVERRIDE SUPPORT
# ======================================================================

def _parse_cli_overrides(args: List[str]) -> Dict[str, Any]:
    """Parse --key value pairs from command line for fine-tuning without editing the file."""
    overrides: Dict[str, Any] = {}
    i = 0
    while i < len(args):
        if args[i] == "--resume" and i + 1 < len(args):
            overrides["resume_from_checkpoint"] = args[i + 1]
            i += 2
        elif args[i].startswith("--") and i + 1 < len(args):
            key = args[i][2:]  # strip --
            val_str = args[i + 1]
            # Type coercion
            if key in ("max_updates", "num_envs", "steps_per_env", "ppo_epochs",
                       "minibatch_size", "gat_layers", "transformer_layers",
                       "transformer_heads", "lstm_layers", "checkpoint_frequency",
                       "n_perturbed_instances", "seed"):
                overrides[key] = int(val_str)
            elif key in ("learning_rate", "gamma", "gae_lambda", "clip_epsilon",
                         "target_kl", "entropy_coefficient", "value_coefficient",
                         "perturbation_fraction"):
                overrides[key] = float(val_str)
            elif key in ("apply_transittime_revision",):
                overrides[key] = val_str.lower() in ("true", "1", "yes")
            else:
                overrides[key] = val_str
            i += 2
        else:
            i += 1
    return overrides


# ======================================================================
# CONFIGURATION VALIDATION
# ======================================================================

def validate_manual_config(
    instance: str,
    policy: str,
    *,
    hidden_dim: int,
    gat_layers: int,
    transformer_layers: int,
    transformer_heads: int,
    lstm_layers: int,
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    ppo_epochs: int,
    clip_epsilon: float,
    target_kl: float,
    entropy_coefficient: float,
    value_coefficient: float,
    num_envs: int,
    steps_per_env: int,
    minibatch_size: int,
    max_updates: int,
    perturbation_fraction: float,
    n_perturbed_instances: int,
    apply_transittime_revision: bool,
    seed: int,
    checkpoint_frequency: int,
    preset: str,
) -> Dict[str, Any]:
    """
    Validate ALL configuration before any PyTorch execution.

    Returns validated config dict. Raises ValueError on invalid config.
    """
    errors: List[str] = []

    # --- Instance / Policy ---
    try:
        resolved_instance = validate_instance(instance)
    except RunnerConfigError as e:
        errors.append(str(e))
        resolved_instance = instance

    try:
        resolved_policy = validate_policy(policy)
    except RunnerConfigError as e:
        errors.append(str(e))
        resolved_policy = policy

    # --- Structural invariants ---
    checks = [
        (hidden_dim > 0, f"hidden_dim must be > 0, got {hidden_dim}"),
        (gat_layers > 0, f"gat_layers must be > 0, got {gat_layers}"),
        (transformer_layers > 0, f"transformer_layers must be > 0, got {transformer_layers}"),
        (transformer_heads > 0, f"transformer_heads must be > 0, got {transformer_heads}"),
        (lstm_layers > 0, f"lstm_layers must be > 0, got {lstm_layers}"),
        (learning_rate > 0, f"learning_rate must be > 0, got {learning_rate}"),
        (0 < gamma <= 1, f"gamma must be in (0, 1], got {gamma}"),
        (0 <= gae_lambda <= 1, f"gae_lambda must be in [0, 1], got {gae_lambda}"),
        (clip_epsilon > 0, f"clip_epsilon must be > 0, got {clip_epsilon}"),
        (minibatch_size > 0, f"minibatch_size must be > 0, got {minibatch_size}"),
        (steps_per_env > 0, f"steps_per_env must be > 0, got {steps_per_env}"),
        (num_envs > 0, f"num_envs must be > 0, got {num_envs}"),
        (max_updates > 0, f"max_updates must be > 0, got {max_updates}"),
        (perturbation_fraction >= 0, f"perturbation_fraction must be >= 0, got {perturbation_fraction}"),
        (entropy_coefficient >= 0, f"entropy_coefficient must be >= 0, got {entropy_coefficient}"),
        (value_coefficient >= 0, f"value_coefficient must be >= 0, got {value_coefficient}"),
        (checkpoint_frequency > 0, f"checkpoint_frequency must be > 0, got {checkpoint_frequency}"),
    ]

    # Divisibility check AFTER confirming heads > 0
    if transformer_heads > 0 and hidden_dim % transformer_heads != 0:
        errors.append(
            f"hidden_dim ({hidden_dim}) must be divisible by "
            f"transformer_heads ({transformer_heads})"
        )

    if perturbation_fraction > 0 and n_perturbed_instances <= 0:
        errors.append(
            f"perturbation_fraction={perturbation_fraction} > 0 but "
            f"n_perturbed_instances={n_perturbed_instances} <= 0"
        )

    for ok, msg in checks:
        if not ok:
            errors.append(msg)

    # --- PAPER mode invariant enforcement ---
    if preset.upper() == "PAPER":
        paper_checks = [
            (hidden_dim == PAPER_HIDDEN_DIM,
             f"PAPER mode requires hidden_dim={PAPER_HIDDEN_DIM}, got {hidden_dim}"),
            (gat_layers == PAPER_GAT_LAYERS,
             f"PAPER mode requires gat_layers={PAPER_GAT_LAYERS}, got {gat_layers}"),
            (transformer_layers == PAPER_TRANSFORMER_LAYERS,
             f"PAPER mode requires transformer_layers={PAPER_TRANSFORMER_LAYERS}, "
             f"got {transformer_layers}"),
            (transformer_heads == PAPER_TRANSFORMER_HEADS,
             f"PAPER mode requires transformer_heads={PAPER_TRANSFORMER_HEADS}, "
             f"got {transformer_heads}"),
            (not apply_transittime_revision,
             "PAPER mode requires apply_transittime_revision=False"),
        ]
        for ok, msg in paper_checks:
            if not ok:
                errors.append("[PAPER MODE] " + msg)

    if errors:
        raise ValueError("Configuration validation failed:\n  " + "\n  ".join(errors))

    return {
        "instance": resolved_instance,
        "policy": resolved_policy,
        "seed": seed,
        "max_updates": max_updates,
        "num_envs": num_envs,
        "steps_per_env": steps_per_env,
        "minibatch_size": minibatch_size,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ppo_epochs": ppo_epochs,
        "clip_epsilon": clip_epsilon,
        "target_kl": target_kl,
        "entropy_coefficient": entropy_coefficient,
        "value_coefficient": value_coefficient,
        "hidden_dim": hidden_dim,
        "gat_layers": gat_layers,
        "transformer_layers": transformer_layers,
        "transformer_heads": transformer_heads,
        "lstm_layers": lstm_layers,
        "checkpoint_frequency": checkpoint_frequency,
        "perturbation_fraction": perturbation_fraction,
        "n_perturbed_instances": n_perturbed_instances,
        "apply_transittime_revision": apply_transittime_revision,
        "preset": preset,
    }


# ======================================================================
# CONFIGURATION DISPLAY
# ======================================================================

def print_active_configuration(cfg: Dict[str, Any], arch_cfg: Any) -> None:
    """Print the COMPLETE ACTIVE TRAINING CONFIGURATION at startup."""
    _border()
    print("  ACTIVE TRAINING CONFIGURATION")
    _border()

    _section("INSTANCE & POLICY")
    print(f"  Instance   : {cfg['instance']}")
    print(f"  Policy     : {cfg['policy']}")
    print(f"  Mode       : {cfg['preset']} ({_get_mode_description(cfg['preset'])})")
    print(f"  Seed       : {cfg['seed']}")
    print()

    _section("ARCHITECTURE")
    print(f"  Hidden dim      : {arch_cfg.hidden_dim}")
    print(f"  GAT layers      : {arch_cfg.gat_layers}")
    print(f"  Transformer layers: {arch_cfg.transformer_layers}")
    print(f"  Transformer heads : {arch_cfg.transformer_heads}")
    print(f"  LSTM layers     : {arch_cfg.lstm_layers}")
    print(f"  Paper-faithful  : {arch_cfg.matches_paper()}")
    print()

    _section("PPO")
    print(f"  Learning rate : {cfg['learning_rate']}")
    print(f"  Gamma         : {cfg['gamma']}")
    print(f"  GAE lambda    : {cfg['gae_lambda']}")
    print(f"  PPO epochs    : {cfg['ppo_epochs']}")
    print(f"  Minibatch size: {cfg['minibatch_size']}")
    print(f"  Clip epsilon  : {cfg['clip_epsilon']}")
    print(f"  Target KL     : {cfg['target_kl']}")
    print(f"  Entropy coef  : {cfg['entropy_coefficient']}")
    print(f"  Value coef    : {cfg['value_coefficient']}")
    print()

    _section("ENVIRONMENT")
    print(f"  Num envs      : {cfg['num_envs']}")
    print(f"  Steps/env     : {cfg['steps_per_env']}")
    print(f"  Max updates   : {cfg['max_updates']}")
    print()

    _section("PERTURBATION")
    print(f"  Fraction       : {cfg['perturbation_fraction']}")
    print(f"  Instances      : {cfg['n_perturbed_instances']}")
    print()

    _section("SEMANTICS")
    print(f"  Transit time   : {'ENABLED' if cfg['apply_transittime_revision'] else 'DISABLED (paper)'}")
    print(f"  Checkpoint freq: every {cfg['checkpoint_frequency']} updates")
    print()
    _border()


# ======================================================================
# EXPERIMENT DIRECTORY
# ======================================================================

def setup_experiment_directory(
    experiment_name: str,
    cfg: Dict[str, Any],
    arch_cfg: Any,
) -> Path:
    """Create a unique experiment directory with timestamp. Does not overwrite."""
    exp_base = _ROOT / "experiments" / "manual"
    exp_base.mkdir(parents=True, exist_ok=True)

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    exp_dir = exp_base / f"{experiment_name}_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot (use .get for optional fields)
    config_snapshot = {
        "experiment_name": experiment_name,
        "timestamp_utc": ts + "Z",
        "mode": cfg.get("preset", "MANUAL"),
        "instance": cfg.get("instance", "Baltic"),
        "policy": cfg.get("policy", "encoder_only"),
        "seed": cfg.get("seed", 42),
        "architecture": arch_cfg.to_dict(),
        "ppo": {
            "learning_rate": cfg.get("learning_rate", 2e-4),
            "gamma": cfg.get("gamma", 1.0),
            "gae_lambda": cfg.get("gae_lambda", 0.9),
            "ppo_epochs": cfg.get("ppo_epochs", 2),
            "minibatch_size": cfg.get("minibatch_size", 32),
            "clip_epsilon": cfg.get("clip_epsilon", 0.2),
            "target_kl": cfg.get("target_kl", 0.1),
            "entropy_coefficient": cfg.get("entropy_coefficient", 0.05),
            "value_coefficient": cfg.get("value_coefficient", 0.5),
        },
        "environment": {
            "num_envs": cfg.get("num_envs", 1),
            "steps_per_env": cfg.get("steps_per_env", 50),
            "max_updates": cfg.get("max_updates", 10),
            "checkpoint_frequency": cfg.get("checkpoint_frequency", 100),
        },
        "perturbation": {
            "fraction": cfg.get("perturbation_fraction", 0.0),
            "n_instances": cfg.get("n_perturbed_instances", 0),
        },
        "semantics": {
            "apply_transit_time_revision": cfg.get("apply_transittime_revision", False),
        },
    }
    write_json(str(exp_dir / "config.json"), config_snapshot)

    return exp_dir


# ======================================================================
# MAIN
# ======================================================================

def main(cli_resume: str = "") -> int:
    import argparse
    # Also parse sys.argv for --resume (allows direct script invocation)
    for arg in sys.argv[1:]:
        if arg == "--resume" and len(sys.argv) > sys.argv.index(arg) + 1:
            idx = sys.argv.index(arg)
            cli_resume = sys.argv[idx + 1]
            break

    parser = argparse.ArgumentParser(
        description="RL Liner Shipping — Manual Training Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--resume", metavar="PATH",
        help="Path to .pt checkpoint to resume training from",
    )
    parser.add_argument(
        "--experiment", metavar="NAME", default=None,
        help="Experiment name (default: auto-generated from preset+instance)",
    )
    cli = parser.parse_args()

    # Apply preset
    _apply_preset(PRESET)

    # Apply CLI overrides (highest priority)
    cli_overrides = _parse_cli_overrides(sys.argv[1:])
    if cli.resume:
        cli_overrides["resume_from_checkpoint"] = cli.resume

    # Override specific fields from CLI
    for key, value in cli_overrides.items():
        if key == "resume_from_checkpoint":
            global RESUME_FROM_CHECKPOINT
            RESUME_FROM_CHECKPOINT = value
        elif key in globals():
            globals()[key] = value

    # ---- Re-read live values after preset/CLI ----
    instance = INSTANCE
    policy = POLICY
    seed = SEED
    max_updates = MAX_UPDATES
    num_envs = NUM_ENVS
    steps_per_env = STEPS_PER_ENV
    minibatch_size = MINIBATCH_SIZE
    learning_rate = LEARNING_RATE
    gamma = GAMMA
    gae_lambda = GAE_LAMBDA
    ppo_epochs = PPO_EPOCHS
    clip_epsilon = CLIP_EPSILON
    target_kl = TARGET_KL
    entropy_coefficient = ENTROPY_COEFFICIENT
    value_coefficient = VALUE_COEFFICIENT
    hidden_dim = HIDDEN_DIM
    gat_layers = GAT_LAYERS
    transformer_layers = TRANSFORMER_LAYERS
    transformer_heads = TRANSFORMER_HEADS
    lstm_layers = LSTM_LAYERS
    checkpoint_frequency = CHECKPOINT_FREQUENCY
    perturbation_fraction = PERTURBATION_FRACTION
    n_perturbed_instances = N_PERTURBED_INSTANCES
    apply_transittime_revision = APPLY_TRANSITTIME_REVISION
    preset = PRESET

    # ---- Validate ----
    try:
        cfg = validate_manual_config(
            instance=instance,
            policy=policy,
            hidden_dim=hidden_dim,
            gat_layers=gat_layers,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            lstm_layers=lstm_layers,
            learning_rate=learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_epochs=ppo_epochs,
            clip_epsilon=clip_epsilon,
            target_kl=target_kl,
            entropy_coefficient=entropy_coefficient,
            value_coefficient=value_coefficient,
            num_envs=num_envs,
            steps_per_env=steps_per_env,
            minibatch_size=minibatch_size,
            max_updates=max_updates,
            perturbation_fraction=perturbation_fraction,
            n_perturbed_instances=n_perturbed_instances,
            apply_transittime_revision=apply_transittime_revision,
            seed=seed,
            checkpoint_frequency=checkpoint_frequency,
            preset=preset,
        )
    except ValueError as e:
        print(f"\nCONFIGURATION VALIDATION FAILED:\n{e}")
        return 1

    # ---- Architecture config ----
    from neural.config import ArchitectureConfig
    arch_cfg = ArchitectureConfig(
        hidden_dim=cfg["hidden_dim"],
        gat_layers=cfg["gat_layers"],
        transformer_layers=cfg["transformer_layers"],
        transformer_heads=cfg["transformer_heads"],
        lstm_layers=cfg["lstm_layers"],
    )

    # ---- Print active configuration ----
    print_active_configuration(cfg, arch_cfg)

    # ---- Experiment directory ----
    exp_name = cli.experiment or f"{preset.lower()}_{cfg['instance']}_{cfg['policy']}"
    exp_dir = setup_experiment_directory(exp_name, cfg, arch_cfg)
    print(f"\n  Experiment directory: {exp_dir}")
    print()

    # ---- Load instance ----
    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader("data")
    instance_obj = loader.load(cfg["instance"])

    # Check transit-time revision setting
    if apply_transittime_revision:
        print("  [WARN] Transit-time revision ENABLED (paper excludes this)")
    else:
        print("  [OK] Transit-time revision DISABLED (paper-faithful)")
    print()

    data_hashes = collect_data_hashes(cfg["instance"])
    print(f"  Data integrity: {len(data_hashes)} files hashed")

    # ---- Initialize trainer ----
    from policies.training import LinerShippingTrainer, TrainingConfig

    training_config = TrainingConfig(
        dataset=cfg["instance"],
        policy=cfg["policy"],
        learning_rate=cfg["learning_rate"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        ppo_epochs=cfg["ppo_epochs"],
        clip_epsilon=cfg["clip_epsilon"],
        target_kl=cfg["target_kl"],
        entropy_coefficient=cfg["entropy_coefficient"],
        value_coefficient=cfg["value_coefficient"],
        num_envs=cfg["num_envs"],
        steps_per_env=cfg["steps_per_env"],
        minibatch_size=cfg["minibatch_size"],
        seed=cfg["seed"],
        max_updates=cfg["max_updates"],
        checkpoint_frequency=cfg["checkpoint_frequency"],
        hidden_dim=cfg["hidden_dim"],
        gat_layers=cfg["gat_layers"],
        transformer_layers=cfg["transformer_layers"],
        transformer_heads=cfg["transformer_heads"],
        lstm_layers=cfg["lstm_layers"],
        perturbation_fraction=cfg["perturbation_fraction"],
        n_perturbed_instances=cfg["n_perturbed_instances"],
    )

    ckpt_dir = _ROOT / "checkpoints" / f"manual_{cfg['instance']}_{cfg['policy']}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    trainer = LinerShippingTrainer(
        instance_name=cfg["instance"],
        policy_type=cfg["policy"],
        config=training_config,
        checkpoint_dir=str(ckpt_dir),
    )

    # ---- Resume from checkpoint if requested ----
    resume_path = RESUME_FROM_CHECKPOINT or cli_resume
    if resume_path:
        if os.path.exists(resume_path):
            try:
                trainer.load_checkpoint(resume_path)
                print(f"\n  Resumed from checkpoint: {resume_path}")
                print(f"  Current update count: {trainer._update_count}")
                print(f"  Continuing for {cfg['max_updates']} total updates...")
            except Exception as e:
                print(f"\n  ERROR loading checkpoint: {e}")
                print("  Starting fresh instead.")
                trainer = LinerShippingTrainer(
                    instance_name=cfg["instance"],
                    policy_type=cfg["policy"],
                    config=training_config,
                    checkpoint_dir=str(ckpt_dir),
                )
        else:
            print(f"\n  WARNING: Resume path '{resume_path}' does not exist. Starting fresh.")

    print(f"\n  Trainer initialized.")
    print(f"  Checkpoint dir: {ckpt_dir}")

    # ---- Run training ----
    _section("TRAINING RUN")
    train_start = time.time()
    metrics = trainer.run_training(max_updates=cfg["max_updates"])
    train_time = time.time() - train_start
    summary = trainer.get_summary()

    # ---- Validation ----
    _section("VALIDATION")
    last_metric = metrics[-1] if metrics else None
    result_dict = {
        "final_eta": summary.get("final_profit", 0.0),
        "revenue": abs(summary.get("final_profit", 0.0)),  # eta is negative profit
        "C_reject": 0.0,
        "C_handle": 0.0,
        "C_service": 0.0,
        "C_unused": 0.0,
        "C_voyage": 0.0,
        "routed_demand": 0.0,
        "rejected_demand": 0.0,
        "total_services": summary.get("episodes", 0),
        "policy_type": cfg["policy"],
        "training_updates": summary.get("updates", 0),
        "policy_loss": last_metric.PPO_policy_loss if last_metric else 0.0,
        "value_loss": last_metric.PPO_value_loss if last_metric else 0.0,
        "approx_kl": last_metric.PPO_approx_kl if last_metric else 0.0,
        "gradient_norm": last_metric.gradient_norm if last_metric else 0.0,
    }

    checks = validate_result(result_dict, check_services=False, check_structure=False)
    passed_checks = {k: v for k, v in checks.items() if v != VALIDATION_NA}
    all_passed = all(v == VALIDATION_PASSED for v in passed_checks.values())

    print(f"\n  Numerical checks:")
    for name, check in checks.items():
        status = "PASS" if check == VALIDATION_PASSED else "FAIL"
        print(f"    [{status}] {name}")

    if last_metric:
        finite_checks = {
            "policy_loss_finite": math.isfinite(last_metric.PPO_policy_loss),
            "value_loss_finite": math.isfinite(last_metric.PPO_value_loss),
            "entropy_finite": math.isfinite(last_metric.PPO_entropy),
            "kl_finite": math.isfinite(last_metric.PPO_approx_kl),
            "grad_norm_finite": math.isfinite(last_metric.gradient_norm),
        }
        print(f"\n  Loss finiteness:")
        for name, ok in finite_checks.items():
            status = "PASS" if ok else "FAIL"
            print(f"    [{status}] {name}")

    # ---- Save final checkpoint ----
    _section("CHECKPOINT")
    ckpt_path = trainer.save_checkpoint("final_checkpoint.pt")
    print(f"  Final checkpoint saved: {ckpt_path}")

    # ---- Save experiment artifacts ----
    _section("EXPERIMENT ARTIFACTS")

    repro_info = collect_reproducibility_info(cfg)
    repro_info["checkpoint_path"] = ckpt_path
    repro_info["data_hashes"] = data_hashes
    repro_info["experiment_dir"] = str(exp_dir)
    repro_info["train_time_seconds"] = train_time

    # Save training metrics CSV
    metrics_csv = exp_dir / "training_metrics.csv"
    if metrics:
        fieldnames = [
            "update", "episode", "reward", "normalized_reward", "episode_return",
            "network_profit_eta", "num_services", "rejected_demand",
            "C_service", "C_unused", "C_voyage", "C_reject", "C_handle",
            "PPO_policy_loss", "PPO_value_loss", "PPO_entropy",
            "PPO_approx_kl", "PPO_clip_fraction", "PPO_advantage_mean",
            "PPO_advantage_std", "PPO_value_mean", "PPO_value_std",
            "gradient_norm", "wall_clock_time", "vessel_usage",
        ]
        with open(metrics_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics:
                row = {k: getattr(m, k, None) for k in fieldnames}
                # Convert vessel_usage dict to string
                if hasattr(m, "vessel_usage") and m.vessel_usage:
                    row["vessel_usage"] = json.dumps(m.vessel_usage)
                writer.writerow(row)
        print(f"  Metrics CSV: {metrics_csv}")

    # Save training metrics JSONL (one record per update)
    metrics_jsonl = exp_dir / "training_metrics.jsonl"
    with open(metrics_jsonl, "w", encoding="utf-8") as f:
        for m in metrics:
            d = m.to_dict()
            f.write(json.dumps(d, default=str) + "\n")
    print(f"  Metrics JSONL: {metrics_jsonl}")

    # Save experiment result JSON
    training_result = {
        "experiment": {
            "name": exp_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": cfg["preset"],
            "experiment_dir": str(exp_dir),
        },
        "configuration": {
            "instance": cfg["instance"],
            "policy": cfg["policy"],
            "seed": cfg["seed"],
            "max_updates": cfg["max_updates"],
            "num_envs": cfg["num_envs"],
            "steps_per_env": cfg["steps_per_env"],
            "minibatch_size": cfg["minibatch_size"],
            "architecture": arch_cfg.to_dict(),
            "ppo": {
                "learning_rate": cfg["learning_rate"],
                "gamma": cfg["gamma"],
                "gae_lambda": cfg["gae_lambda"],
                "ppo_epochs": cfg["ppo_epochs"],
                "clip_epsilon": cfg["clip_epsilon"],
                "target_kl": cfg["target_kl"],
                "entropy_coefficient": cfg["entropy_coefficient"],
                "value_coefficient": cfg["value_coefficient"],
            },
            "perturbation": {
                "fraction": cfg["perturbation_fraction"],
                "n_instances": cfg["n_perturbed_instances"],
            },
            "semantics": {
                "apply_transit_time_revision": cfg["apply_transittime_revision"],
            },
        },
        "training_summary": summary,
        "training_time_seconds": train_time,
        "validation": dict(checks),
        "reproducibility": repro_info,
        "status": "success" if all_passed else "validation_issues",
    }
    result_file = exp_dir / "summary.json"
    write_json(str(result_file), training_result)
    print(f"  Summary JSON:  {result_file}")

    # ---- Training summary ----
    print(f"\n{'='*60}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"  Updates     : {summary.get('updates', 0)}")
    print(f"  Episodes    : {summary.get('episodes', 0)}")
    print(f"  Final eta     : {summary.get('final_profit', 'N/A'):,.2f}")
    print(f"  Best eta      : {summary.get('max_reward', summary.get('min_reward', 'N/A')):.4f}")
    print(f"  Mean reward : {summary.get('mean_reward', 'N/A'):.4f}")
    print(f"  Final KL    : {summary.get('final_kl', 'N/A'):.6f}")
    print(f"  Final entropy: {summary.get('final_entropy', 'N/A'):.4f}")
    print(f"  Wall clock  : {train_time:.1f}s")
    print(f"{'='*60}")

    # ---- Classification ----
    if not metrics:
        classification = "PIPELINE_FAILURE"
    elif summary.get("final_profit", 0) > -1e6:
        classification = "SUCCESSFUL_LEARNING"
    elif summary.get("final_kl", 1.0) < 0.001 and summary.get("updates", 0) > 5:
        classification = "EXPLORATION_COLLAPSE"
    else:
        classification = "INSUFFICIENT_TRAINING"

    print(f"\n  Initial classification: {classification}")
    print(f"  (See MANUAL_TRAINING_REPORT.md after detailed analysis)")
    print()

    _border()
    print(f"  STATUS: {'SUCCESS' if all_passed else 'VALIDATION ISSUES'}")
    print(f"  Experiment artifacts: {exp_dir}/")
    print(f"  Checkpoints: {ckpt_dir}/")
    _border()
    print()

    return 0 if all_passed else 1


if __name__ == "__main__":
    # Parse --resume from sys.argv before argparse runs (avoids conflict)
    cli_resume = ""
    argv = sys.argv[1:]
    for i, arg in enumerate(argv, 0):
        if arg == "--resume" and i + 1 < len(argv):
            cli_resume = argv[i + 1]
            break
    sys.exit(main(cli_resume=cli_resume))
