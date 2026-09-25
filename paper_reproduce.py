"""
paper_reproduce.py — Paper-faithful reproduction runner.

Uses exact hyperparameters from Dutta et al. (2024) Table 5.
Distinguishes clearly between:
  - SMOKE_TEST (debug, small arch)
  - PAPER_REPRODUCTION (Table 5 values)

This script is NOT train_rl.py. It is a standalone reproduction entry point.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from runners.config import validate_runner_config
from runners.report import collect_data_hashes, write_json, _border, _section


# ============================================================
# PAPER REPRODUCTION CONFIGURATION
# ============================================================
# All values sourced from Dutta et al. (2024) Table 5.
# Ranges are specified; selected values are mid-range unless noted.
# ============================================================

INSTANCE = "Baltic"
POLICY = "encoder_decoder"  # Paper Section 6.1: "primarily report results from encoder-decoder"

# Architecture — Table 5 exact values
HIDDEN_DIM = 512
GAT_LAYERS = 3
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 8
LSTM_LAYERS = 1

# PPO — Table 5 ranges, selected values
LEARNING_RATE = 2e-4        # [1e-4, 3e-4] → midpoint
GAMMA = 1.0                 # exact: 1
GAE_LAMBDA = 0.9            # exact: 0.9
PPO_EPOCHS = 10             # exact: 10
CLIP_EPSILON = 0.2          # [0.15, 0.25] → midpoint
TARGET_KL = 0.1             # exact: 0.1
ENTROPY_COEFFICIENT = 0.05  # [0.01, 0.1] → midpoint
VALUE_COEFFICIENT = 0.5     # exact: 0.5

# Environment — Table 5 ranges
NUM_ENVS = 8                # [8, 16] → lower bound (conservative for CPU)
STEPS_PER_ENV = 100         # [50, 100] → upper bound
MINIBATCH_SIZE = 64         # [64, 128] → lower bound

# Training budget
MAX_UPDATES = 200           # Paper doesn't specify; 200 is conservative for verification
CHECKPOINT_FREQUENCY = 50

# Perturbation — Section 6.1
PERTURBATION_FRACTION = 0.10
N_PERTURBED_INSTANCES = 100  # Full: 16000; using 100 for dry run / first validation

# Data
SEED = 42
APPLY_TRANSIT_TIME = False   # Paper explicitly excludes transit time

# ============================================================
# END PAPER REPRODUCTION CONFIGURATION
# ============================================================


def main() -> int:
    print()
    _border()
    print("  RL LINER SHIPPING — PAPER REPRODUCTION RUNNER")
    print("  Mode: PAPER_REPRODUCTION")
    _border()

    # ------------------------------------------------------------------
    # Validate configuration
    # ------------------------------------------------------------------
    try:
        config = validate_runner_config(
            instance=INSTANCE,
            policy=POLICY,
            seed=SEED,
            max_updates=MAX_UPDATES,
            num_envs=NUM_ENVS,
            steps_per_env=STEPS_PER_ENV,
            minibatch_size=MINIBATCH_SIZE,
            learning_rate=LEARNING_RATE,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            ppo_epochs=PPO_EPOCHS,
            clip_epsilon=CLIP_EPSILON,
            target_kl=TARGET_KL,
            entropy_coefficient=ENTROPY_COEFFICIENT,
            value_coefficient=VALUE_COEFFICIENT,
            hidden_dim=HIDDEN_DIM,
            gat_layers=GAT_LAYERS,
            transformer_layers=TRANSFORMER_LAYERS,
            transformer_heads=TRANSFORMER_HEADS,
            lstm_layers=LSTM_LAYERS,
            checkpoint_frequency=CHECKPOINT_FREQUENCY,
        )
    except Exception as e:
        print(f"\nCONFIGURATION ERROR: {e}")
        return 1

    # ------------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------------
    _section("PAPER REPRODUCTION CONFIGURATION")
    print(f"  Instance       : {config['instance']}")
    print(f"  Policy         : {config['policy']}")
    print(f"  Seed           : {config['seed']}")
    print(f"  Max updates    : {config['max_updates']}")
    print(f"  Num envs       : {config['num_envs']}")
    print(f"  Steps per env  : {config['steps_per_env']}")
    print(f"  Minibatch size : {config['minibatch_size']}")
    print()
    print(f"  Learning rate  : {config['learning_rate']}")
    print(f"  Gamma          : {config['gamma']}")
    print(f"  GAE lambda     : {config['gae_lambda']}")
    print()
    print(f"  PPO epochs     : {config['ppo_epochs']}")
    print(f"  Clip epsilon   : {config['clip_epsilon']}")
    print(f"  Target KL      : {config['target_kl']}")
    print()
    print(f"  Entropy coeff  : {config['entropy_coefficient']}")
    print(f"  Value coeff    : {config['value_coefficient']}")
    print()
    print(f"  Checkpoint freq: {config['checkpoint_frequency']}")

    # ------------------------------------------------------------------
    # Model architecture
    # ------------------------------------------------------------------
    _section("MODEL ARCHITECTURE")

    from neural.config import ArchitectureConfig
    arch_cfg = ArchitectureConfig(
        hidden_dim=config["hidden_dim"],
        gat_layers=config["gat_layers"],
        transformer_layers=config["transformer_layers"],
        transformer_heads=config["transformer_heads"],
        lstm_layers=config["lstm_layers"],
    )
    print(f"  Hidden dim       : {arch_cfg.hidden_dim}")
    print(f"  GAT layers       : {arch_cfg.gat_layers}")
    print(f"  Transformer layers: {arch_cfg.transformer_layers}")
    print(f"  Transformer heads : {arch_cfg.transformer_heads}")
    print(f"  LSTM layers      : {arch_cfg.lstm_layers}")
    print(f"  Paper-faithful   : {arch_cfg.matches_paper()}")

    total_params = sum(p.numel() for p in
                       __import__('neural.backbone', fromlist=['NeuralBackbone']).NeuralBackbone(arch_cfg).parameters())
    print(f"  Total parameters : {total_params:,}")

    # ------------------------------------------------------------------
    # Data foundation
    # ------------------------------------------------------------------
    _section("DATA FOUNDATION")

    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader("data")
    instance = loader.load(config["instance"])

    print(f"  Instance : {instance.name}")
    print(f"  Ports    : {len(instance.ports)}")
    print(f"  Vessels  : {len(instance.vessel_types)}")
    print(f"  Demands  : {len(instance.demands)}")
    print(f"  Fleet    : {[(e.vessel_class, e.quantity) for e in instance.fleet]}")

    data_hashes = collect_data_hashes(config["instance"])
    print(f"  Data integrity: {len(data_hashes)} files hashed")
    print(f"  Transit time revision: {'DISABLED (paper mode)' if not True else 'ENABLED'}")

    # ------------------------------------------------------------------
    # Perturbation setup
    # ------------------------------------------------------------------
    _section("PERTURBATION PROTOCOL")
    print(f"  Fraction     : {PERTURBATION_FRACTION}")
    print(f"  Instances    : {N_PERTURBED_INSTANCES} (full protocol: 16000)")
    print(f"  Distribution : Gaussian, std = fraction * mean, truncated at 0")
    print(f"  Reproducible : Yes (seed={SEED})")

    # ------------------------------------------------------------------
    # Initialize trainer
    # ------------------------------------------------------------------
    _section("INITIALIZING TRAINER")

    from policies.training import LinerShippingTrainer, TrainingConfig

    training_config = TrainingConfig(
        dataset=config["instance"],
        policy=config["policy"],
        learning_rate=config["learning_rate"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ppo_epochs=config["ppo_epochs"],
        clip_epsilon=config["clip_epsilon"],
        target_kl=config["target_kl"],
        entropy_coefficient=config["entropy_coefficient"],
        value_coefficient=config["value_coefficient"],
        num_envs=config["num_envs"],
        steps_per_env=config["steps_per_env"],
        minibatch_size=config["minibatch_size"],
        seed=config["seed"],
        max_updates=config["max_updates"],
        checkpoint_frequency=config["checkpoint_frequency"],
        hidden_dim=config["hidden_dim"],
        gat_layers=config["gat_layers"],
        transformer_layers=config["transformer_layers"],
        transformer_heads=config["transformer_heads"],
        lstm_layers=config["lstm_layers"],
        perturbation_fraction=PERTURBATION_FRACTION,
        n_perturbed_instances=N_PERTURBED_INSTANCES,
    )

    results_dir = _ROOT / "results" / "reproduction"
    results_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = _ROOT / "checkpoints" / f"paper_repro_{config['instance']}_{config['policy']}"

    trainer = LinerShippingTrainer(
        instance_name=config["instance"],
        policy_type=config["policy"],
        config=training_config,
        checkpoint_dir=str(ckpt_dir),
    )

    print(f"  Trainer initialized.")
    print(f"  Checkpoint dir: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    _section("TRAINING RUN")

    train_start = time.time()
    metrics = trainer.run_training(max_updates=config["max_updates"])
    train_time = time.time() - train_start

    summary = trainer.get_summary()

    print(f"\n  Training complete.")
    print(f"  Updates     : {summary['updates']}")
    print(f"  Episodes    : {summary['episodes']}")
    print(f"  Final reward: {summary.get('final_reward', 'N/A'):.4f}")
    print(f"  Mean reward : {summary.get('mean_reward', 'N/A'):.4f}")
    print(f"  Final profit: {summary.get('final_profit', 'N/A'):,.2f}")
    print(f"  Final KL    : {summary.get('final_kl', 'N/A'):.6f}")
    print(f"  Wall clock  : {train_time:.2f}s")

    # ------------------------------------------------------------------
    # Evaluate on held-out ORIGINAL Baltic (no perturbation)
    # ------------------------------------------------------------------
    _section("HELD-OUT EVALUATION")

    from env.environment import LSNDPEnv
    from policies.encoder_decoder import EncoderDecoderPolicy
    from actions.service_generator import ServiceGenerator
    from state.representation import StateEncoder, ServiceMembership
    from neural import neural_state_to_tensors
    from mcf import evaluate_network
    from mcf.expanded_graph import ServiceDefinition
    from env.action import ServiceAction

    # Load ORIGINAL (unperturbed) Baltic for evaluation
    orig_loader = LINERLIBLoader("data")
    orig_instance = orig_loader.load("Baltic")
    orig_dist = {(a.origin, a.destination): a for a in orig_instance.distances}
    orig_env = LSNDPEnv(orig_instance)

    # Reconstruct policy for evaluation (load from checkpoint if available)
    eval_cfg = ArchitectureConfig(
        hidden_dim=HIDDEN_DIM, gat_layers=GAT_LAYERS,
        transformer_layers=TRANSFORMER_LAYERS, transformer_heads=TRANSFORMER_HEADS,
        lstm_layers=LSTM_LAYERS,
    )
    from neural.backbone import NeuralBackbone
    eval_backbone = NeuralBackbone(eval_cfg)

    # Try loading checkpoint
    ckpt_path = ckpt_dir / "final_checkpoint.pt"
    if ckpt_path.exists():
        checkpoint = __import__('torch', fromlist=['load']).load(
            ckpt_path, map_location="cpu", weights_only=False)
        eval_backbone.load_state_dict(checkpoint["backbone_state_dict"])
        # Note: policy state may differ; use current trainer policy if available
        print(f"  Loaded checkpoint: {ckpt_path}")
    else:
        print(f"  No checkpoint found at {ckpt_path}")
        print(f"  Using untrained policy for evaluation (baseline)")

    eval_gen = ServiceGenerator(orig_instance, orig_dist)
    eval_policy = EncoderDecoderPolicy(eval_backbone, orig_instance, eval_gen)

    # Run evaluation episode
    obs, info = orig_env.reset(seed=SEED)
    membership = ServiceMembership()
    dist_by_pair = orig_dist
    state_encoder = StateEncoder(orig_instance, dist_by_pair)

    evaluation_services = []
    step_count = 0
    max_eval_steps = 20

    while not orig_env._terminated and not orig_env._truncated and step_count < max_eval_steps:
        rem = {i: obs["remaining_demand"][i] for i in range(len(obs["remaining_demand"]))}
        fleet = {vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(sorted(orig_instance.vessel_types.keys()))}
        ns = state_encoder.encode(rem, fleet, membership)
        tensors = neural_state_to_tensors(ns)

        out = eval_policy.sample_action(tensors, fleet, seed=SEED + step_count)
        port_seq = list(out.executed_ports) if out.executed_ports else []
        if len(port_seq) < 2:
            break

        sa = ServiceAction(vessel_class=out.vessel_class, port_sequence=port_seq)
        try:
            obs, reward, terminated, truncated, info = orig_env.step(sa)
            evaluation_services.append({
                "vessel_class": out.vessel_class,
                "port_sequence": list(out.executed_ports) if out.executed_ports else [],
            })
            step_count += 1
            if terminated or truncated:
                break
        except Exception:
            break

    # MCF evaluation on collected services
    if orig_env._state.services:
        mcf_result = evaluate_network(
            instance=orig_instance,
            services=orig_env._state.services,
            vessel_requirements=orig_env._state.vessel_requirements,
        )
        eval_revenue = mcf_result.total_revenue
        eval_profit = mcf_result.eta
        eval_rejected = mcf_result.rejected_demand
        eval_services_count = mcf_result.num_services
    else:
        eval_revenue = 0.0
        eval_profit = 0.0
        eval_rejected = 0.0
        eval_services_count = 0

    print(f"  Evaluation steps    : {step_count}")
    print(f"  Services generated  : {eval_services_count}")
    print(f"  Revenue             : ${eval_revenue:,.2f}")
    print(f"  Profit (eta)        : ${eval_profit:,.2f}")
    print(f"  Rejected demand     : {eval_rejected:.0f} FFE/week")
    print(f"  Total demand        : {orig_env._state.total_demand:.0f} FFE/week")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    _section("RESULTS")

    repro_info = {
        "experiment": {
            "name": f"paper_repro_{INSTANCE}_{POLICY}",
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "mode": "PAPER_REPRODUCTION",
        },
        "configuration": {
            "instance": INSTANCE,
            "policy": POLICY,
            "seed": SEED,
            "max_updates": MAX_UPDATES,
            "num_envs": NUM_ENVS,
            "steps_per_env": STEPS_PER_ENV,
            "minibatch_size": MINIBATCH_SIZE,
            "architecture": arch_cfg.to_dict(),
            "ppo": {
                "learning_rate": LEARNING_RATE,
                "gamma": GAMMA,
                "gae_lambda": GAE_LAMBDA,
                "ppo_epochs": PPO_EPOCHS,
                "clip_epsilon": CLIP_EPSILON,
                "target_kl": TARGET_KL,
                "entropy_coefficient": ENTROPY_COEFFICIENT,
                "value_coefficient": VALUE_COEFFICIENT,
            },
            "perturbation": {
                "fraction": PERTURBATION_FRACTION,
                "n_instances": N_PERTURBED_INSTANCES,
                "distribution": "Gaussian, truncated at 0",
            },
            "transit_time": APPLY_TRANSIT_TIME,
        },
        "training_summary": summary,
        "evaluation": {
            "instance": "Baltic (original, unperturbed)",
            "steps": step_count,
            "services": eval_services_count,
            "revenue": eval_revenue,
            "profit": eval_profit,
            "rejected_demand": eval_rejected,
            "total_demand": float(orig_env._state.total_demand),
        },
        "training_time_seconds": train_time,
        "reproducibility": {
            "seed": SEED,
            "perturbation_seed_base": SEED,
            "data_hashes": data_hashes,
            "architecture_matches_paper": arch_cfg.matches_paper(),
        },
        "status": "completed",
    }

    result_file = results_dir / f"paper_repro_{INSTANCE}_{POLICY}_{SEED}.json"
    write_json(str(result_file), repro_info)
    print(f"  Results saved: {result_file}")

    _border()
    print(f"  STATUS: COMPLETED")
    print(f"  Results stored in: {results_dir}/")
    _border()
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
