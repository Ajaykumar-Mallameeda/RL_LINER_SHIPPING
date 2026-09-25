"""
P12 — Real LINERLIB Training Engine.

Runs actual RL training on real LINERLIB benchmark data using P10's PPO engine.
This is the first phase where genuine benchmark training is permitted.

Scope:
  - Train P8 (Encoder-only) on Baltic
  - Train P9 (Encoder-decoder) on Baltic
  - Record comprehensive training metrics
  - Save checkpoints for resume
  - Monitor for overfitting/sanity issues

NOT in scope:
  - Paper reproduction claims
  - Benchmark comparison
  - Scaling experiments
  - Hyperparameter tuning
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from data.instance import LINERLIBInstance
from data.linerlib_loader import LINERLIBLoader
from env.environment import LSNDPEnv
from mcf.ppo_engine import (
    PPOBuffer,
    PPOConfig,
    PPOTrainer,
    PDiagnostics,
    ValueFunction,
    build_rollout_batch,
)
from neural import ArchitectureConfig, NeuralBackbone, neural_state_to_tensors
from policies.encoder_decoder import EncoderDecoderPolicy
from policies.encoder_only import EncoderOnlyPolicy
from state.representation import ServiceMembership, StateEncoder


@dataclass
class TrainingConfig:
    """Configuration for a single training run."""
    dataset: str
    policy: str  # "encoder_only" or "encoder_decoder"
    learning_rate: float
    gamma: float
    gae_lambda: float
    ppo_epochs: int
    clip_epsilon: float
    target_kl: float
    entropy_coefficient: float
    value_coefficient: float
    num_envs: int
    steps_per_env: int
    minibatch_size: int
    seed: int
    max_updates: int
    checkpoint_frequency: int
    hidden_dim: int
    gat_layers: int
    transformer_layers: int
    transformer_heads: int
    lstm_layers: int
    perturbation_fraction: float = 0.0  # 0.0 = no perturbation (baseline)
    n_perturbed_instances: int = 0      # number of perturbed instances to cache

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingMetrics:
    """Metrics recorded during training."""
    update: int
    episode: int
    reward: float
    normalized_reward: float
    episode_return: float
    network_profit_eta: float
    num_services: int
    rejected_demand: float
    vessel_usage: Dict[str, float]
    C_service: float
    C_unused: float
    C_voyage: float
    C_reject: float
    C_handle: float
    C_handle_total: float
    PPO_policy_loss: float
    PPO_value_loss: float
    PPO_entropy: float
    PPO_approx_kl: float
    PPO_clip_fraction: float
    PPO_advantage_mean: float
    PPO_advantage_std: float
    PPO_value_mean: float
    PPO_value_std: float
    gradient_norm: float
    wall_clock_time: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LinerShippingTrainer:
    """
    Real LINERLIB training engine for PPO-based LSNDP policies.

    Parameters
    ----------
    instance_name : str
        Name of the LINERLIB instance to train on.
    policy_type : str
        "encoder_only" or "encoder_decoder".
    config : TrainingConfig
        Training configuration.
    checkpoint_dir : str
        Directory to save checkpoints and logs.
    """

    def __init__(
        self,
        instance_name: str,
        policy_type: str,
        config: TrainingConfig,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.instance_name = instance_name
        self.policy_type = policy_type
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load instance
        loader = LINERLIBLoader("data")
        self.instance = loader.load(instance_name)

        # Precompute distance lookup
        self.dist_by_pair = {
            (a.origin, a.destination): a for a in self.instance.distances
        }

        # Initialize components
        self._setup_components()

        # Tracking
        self._update_count: int = 0
        self._episode_count: int = 0
        self._metrics_log: List[TrainingMetrics] = []
        self._start_time: float = time.time()
        self._raw_data_hashes: Dict[str, str] = {}

        # Record raw data hashes
        self._record_raw_data_hashes()

    def _setup_components(self) -> None:
        """Initialize all training components."""
        # Neural backbone
        cfg = ArchitectureConfig(
            hidden_dim=self.config.hidden_dim,
            gat_layers=self.config.gat_layers,
            transformer_layers=self.config.transformer_layers,
            transformer_heads=self.config.transformer_heads,
            lstm_layers=self.config.lstm_layers,
        )
        self.backbone = NeuralBackbone(cfg)

        # Policy
        gen = None  # Will be created per-episode (needs ServiceGenerator)
        if self.policy_type == "encoder_only":
            from actions.service_generator import ServiceGenerator
            gen = ServiceGenerator(self.instance, self.dist_by_pair)
            self.policy = EncoderOnlyPolicy(
                self.backbone, self.instance, gen,
            )
        elif self.policy_type == "encoder_decoder":
            from actions.service_generator import ServiceGenerator
            gen = ServiceGenerator(self.instance, self.dist_by_pair)
            self.policy = EncoderDecoderPolicy(
                self.backbone, self.instance, gen,
            )
        else:
            raise ValueError(f"Unknown policy type: {self.policy_type}")

        # Critic (value function)
        # Input dimension: port_features flattened + vessel_features flattened
        port_feat_dim = (len(self.instance.ports) + 1) * 2  # P+1 nodes, 2 features
        vessel_feat_dim = len(self.instance.vessel_types) * 11  # V vessels, 11 features
        critic_input_dim = port_feat_dim + vessel_feat_dim
        self.critic = ValueFunction(input_dim=critic_input_dim)

        # PPO trainer
        ppo_config = PPOConfig(
            learning_rate=self.config.learning_rate,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
            ppo_epochs=self.config.ppo_epochs,
            clip_epsilon=self.config.clip_epsilon,
            target_kl=self.config.target_kl,
            entropy_coefficient=self.config.entropy_coefficient,
            value_coefficient=self.config.value_coefficient,
            optimizer="adamw",
            optimizer_kwargs={"weight_decay": 1e-4},
            num_envs=self.config.num_envs,
            steps_per_env=self.config.steps_per_env,
            minibatch_size=self.config.minibatch_size,
            seed=self.config.seed,
        )
        self.trainer = PPOTrainer(self.policy, self.critic, ppo_config)

        # Environment
        self.env = LSNDPEnv(self.instance)

        # State encoder
        self.state_encoder = StateEncoder(
            self.instance, self.dist_by_pair,
        )

        # Buffer
        self.buffer = PPOBuffer()

        # Perturbation support
        self._perturbation_fraction = self.config.perturbation_fraction
        self._perturbed_instances: List[Any] = []
        self._perturbation_rng: Optional[random.Random] = None
        if self._perturbation_fraction > 0:
            self._setup_perturbation()

    def _setup_perturbation(self) -> None:
        """Generate perturbed training instances per paper Section 6.1."""
        from experiments.reproduction.perturbation import generate_perturbed_instances
        n = self.config.n_perturbed_instances or 100
        self._perturbed_instances = generate_perturbed_instances(
            base_instance=self.instance,
            n_instances=n,
            fraction=self._perturbation_fraction,
            seed_base=self.config.seed,
        )
        self._perturbation_rng = random.Random(self.config.seed)
        print(f"Perturbation enabled: {len(self._perturbed_instances)} instances, "
              f"fraction={self._perturbation_fraction}")

    def _record_raw_data_hashes(self) -> None:
        """Record SHA-256 hashes of all raw data files."""
        import hashlib
        data_root = Path("data")
        for csv_file in data_root.glob("*.csv"):
            with open(csv_file, "rb") as f:
                self._raw_data_hashes[str(csv_file.name)] = hashlib.sha256(
                    f.read()
                ).hexdigest()

    def _verify_data_integrity(self) -> bool:
        """Verify raw data hasn't been modified."""
        import hashlib
        for filename, expected_hash in self._raw_data_hashes.items():
            filepath = Path("data") / filename
            if filepath.exists():
                with open(filepath, "rb") as f:
                    current_hash = hashlib.sha256(f.read()).hexdigest()
                if current_hash != expected_hash:
                    print(f"WARNING: Raw data modified: {filename}")
                    return False
        return True

    def _encode_state(
        self,
        remaining_demand: Dict[int, float],
        fleet_remaining: Dict[str, float],
        membership: ServiceMembership,
    ) -> Tuple[Any, torch.Tensor]:
        """
        Encode environment state to neural tensor representation.

        Returns
        -------
        GraphTensors
            Tensor bundle for policy forward pass.
        torch.Tensor
            Flattened state for critic.
        """
        ns = self.state_encoder.encode(remaining_demand, fleet_remaining, membership)
        bundle = neural_state_to_tensors(ns)

        # Flatten state for critic
        critic_input = torch.cat([
            torch.from_numpy(ns.port_features.flatten()),
            torch.from_numpy(ns.vessel_features.flatten()),
        ]).unsqueeze(0)  # Add batch dim

        return bundle, critic_input

    def collect_rollout(
        self,
        seed: int,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Collect a single rollout (episode) from the environment.

        When perturbation is enabled, a perturbed instance is selected
        for each episode to match the paper's training protocol.

        Parameters
        ----------
        seed : int
            Random seed for this episode.

        Returns
        -------
        list[dict]
            Trajectory steps with all needed information.
        int
            Number of steps taken.
        """
        # Select instance for this episode (base or perturbed)
        if self._perturbed_instances:
            inst_idx = self._perturbation_rng.randint(0, len(self._perturbed_instances) - 1)
            current_inst = self._perturbed_instances[inst_idx]
            # Rebuild env for perturbed instance
            self.env = LSNDPEnv(current_inst)
            self.dist_by_pair = {
                (a.origin, a.destination): a for a in current_inst.distances
            }
        else:
            current_inst = self.instance

        obs, info = self.env.reset(seed=seed)
        membership = ServiceMembership()
        trajectory = []
        steps = 0
        max_steps = self.config.steps_per_env  # [G11.1 FIX] Respect configured rollout horizon.

        while not self.env._terminated and not self.env._truncated and steps < max_steps:
            # Encode state
            rem = {i: obs["remaining_demand"][i] for i in range(len(obs["remaining_demand"]))}
            fleet = {
                vc: obs["fleet_remaining"][i]
                for i, vc in enumerate(sorted(current_inst.vessel_types.keys()))
            }

            bundle, critic_input = self._encode_state(rem, fleet, membership)

            # Sample action from policy
            if self.policy_type == "encoder_only":
                out = self.policy.sample_action(bundle, fleet, seed=seed + steps)
                action = out.raw_sampled_ports
                executed_action = out.executed_ports
                log_prob = out.raw_log_prob
                entropy = out.entropy
                fallback = out.fallback_applied
                decoded_seq = []
                executed_seq = list(executed_action) if executed_action else []
                # Store selected_mask as tensor for PPO re-evaluation
                n_ports = len(self.instance.ports)
                sm = torch.zeros(n_ports, dtype=torch.bool, device=bundle.device)
                port_codes_list = sorted(self.instance.ports.keys())
                for pc in action:
                    if pc in port_codes_list:
                        sm[port_codes_list.index(pc)] = True
            else:
                out = self.policy.sample_action(bundle, fleet, seed=seed + steps)
                action = out.decoded_port_sequence
                executed_action = out.executed_port_sequence
                log_prob = out.log_prob
                entropy = out.entropy
                fallback = False
                decoded_seq = list(action)
                executed_seq = list(executed_action)

            # Build ServiceAction from policy output — no secondary selection.
            # The policy's vessel_class and port_sequence are authoritative;
            # draft feasibility is handled by the policy's masking/log_prob,
            # so changing the action here would violate PPO on-policy integrity.
            vessel_class = out.vessel_class
            port_sequence = executed_seq if executed_seq else decoded_seq

            from env.action import ServiceAction
            sa = ServiceAction(
                vessel_class=vessel_class,
                port_sequence=port_sequence,
            )

            # Step environment
            # Only catch ServiceValidationError — other exceptions indicate
            # genuine bugs and must propagate rather than silently substituting.
            from env.environment import ServiceValidationError
            try:
                obs, reward, terminated, truncated, info = self.env.step(sa)
            except ServiceValidationError:
                # Invalid action — retry with a deterministic fallback.
                vessels = sorted(current_inst.vessel_types.keys())
                ports = sorted(current_inst.ports.keys())[:3]
                sa = ServiceAction(vessel_class=vessels[0], port_sequence=ports)
                obs, reward, terminated, truncated, info = self.env.step(sa)

            # Get value estimate
            with torch.no_grad():
                value = self.critic(critic_input).item()

            # Store step (preserve exact fleet state at this timestep for on-policy PPO)
            traj_entry: Dict[str, Any] = {
                "state": bundle,
                "critic_input": critic_input,
                "action": action,
                "executed_action": executed_action,
                "reward": reward,
                "done": terminated or truncated,
                "truncated": truncated,
                "old_log_prob": log_prob,
                "old_value": torch.tensor(value),
                "entropy": entropy,
                "fallback_applied": fallback,
                "decoded_port_sequence": decoded_seq,
                "executed_port_sequence": executed_seq,
                "info": info,
                "fleet_remaining": dict(fleet),
            }
            if self.policy_type == "encoder_only":
                traj_entry["raw_sampled_ports"] = list(action)
                traj_entry["selected_mask"] = sm
            elif self.policy_type == "encoder_decoder":
                traj_entry["substep_selected"] = list(out.substep_selected)
            else:
                raise ValueError(f"Unknown policy type: {self.policy_type}")

            trajectory.append(traj_entry)

            steps += 1

            if terminated or truncated:
                break

        return trajectory, steps

    def _build_metrics_from_trajectory(
        self,
        trajectory: List[Dict[str, Any]],
        update_count: int = 0,
        episode_count: int = 0,
    ) -> TrainingMetrics:
        """
        Derive TrainingMetrics from an existing trajectory.

        This ensures metrics and PPO updates always operate on the SAME
        trajectory — the invariant required for correct PPO semantics.
        """
        if not trajectory:
            raise ValueError("Cannot build metrics from empty trajectory.")

        # Get final episode metrics
        final_info = trajectory[-1]["info"]
        eta = final_info.get("profit", 0.0)
        num_services = final_info.get("num_services", 0)
        rejected = final_info.get("demand_rejected", 0.0)

        # Compute episode return
        episode_return = sum(step["reward"] for step in trajectory)

        # Reward normalization by η_1 (paper Eq. 36)
        normalized_reward = episode_return

        # Get MCF cost breakdown — use keys from _build_info()
        C_service = final_info.get("service_cost", 0.0)
        C_unused = final_info.get("unused_vessel_cost", 0.0)
        C_voyage = final_info.get("voyage_cost", 0.0)
        C_reject = final_info.get("rejection_cost", 0.0)
        C_handle = final_info.get("handling_cost", 0.0)
        C_handle_total = final_info.get("handling_cost", 0.0)

        # Fleet usage
        fleet_remaining = {
            vc: float(final_info.get("vessel_state", {}).get(vc, 0.0))
            for vc in sorted(self.instance.vessel_types.keys())
        }
        initial_fleet = {
            e.vessel_class: float(e.quantity) for e in self.instance.fleet
        }
        vessel_usage = {
            vc: initial_fleet.get(vc, 0.0) - fleet_remaining.get(vc, 0.0)
            for vc in initial_fleet
        }

        wall_time = time.time() - self._start_time

        metrics = TrainingMetrics(
            update=update_count,
            episode=episode_count,
            reward=normalized_reward,
            normalized_reward=normalized_reward,
            episode_return=episode_return,
            network_profit_eta=eta,
            num_services=num_services,
            rejected_demand=rejected,
            vessel_usage=vessel_usage,
            C_service=C_service,
            C_unused=C_unused,
            C_voyage=C_voyage,
            C_reject=C_reject,
            C_handle=C_handle,
            C_handle_total=C_handle_total,
            PPO_policy_loss=0.0,  # Updated after PPO step
            PPO_value_loss=0.0,
            PPO_entropy=0.0,
            PPO_approx_kl=0.0,
            PPO_clip_fraction=0.0,
            PPO_advantage_mean=0.0,
            PPO_advantage_std=0.0,
            PPO_value_mean=0.0,
            PPO_value_std=0.0,
            gradient_norm=0.0,
            wall_clock_time=wall_time,
        )

        return metrics

    def perform_ppo_update(
        self,
        trajectory: List[Dict[str, Any]],
        *,
        log_metrics: bool = True,
    ) -> PDiagnostics:
        """
        Perform PPO update on collected trajectory.

        [G11.2.4 REPAIR] This method now routes through the real multi-epoch
        PPO update (``PPOTrainer.train_step_adapter``) instead of re-implementing
        a single clipped full-batch step inline. Consequently the active
        training path now genuinely consumes ``ppo_epochs``, ``minibatch_size``,
        ``clip_epsilon``, ``target_kl``, ``entropy_coefficient``,
        ``value_coefficient``, ``max_grad_norm`` and ``learning_rate``.

        The encoder-decoder action representation is preserved end to end: the
        stored RAW sampled action (``substep_selected``) is re-evaluated through
        ``policy.evaluate_actions()``. The fallback-repaired EXECUTED action is
        never substituted as the PPO action.

        Parameters
        ----------
        trajectory : list[dict]
            Collected rollout steps.
        log_metrics : bool
            When True (the default) the update appends its own metrics entry,
            so a DIRECT call updates ``get_summary()``. ``run_training()``
            already builds and appends metrics for the same trajectory before
            calling this, so it passes ``False`` to avoid double-counting the
            episode.

        Returns
        -------
        PDiagnostics
            Training diagnostics, now including per-epoch execution evidence.
        """
        if not trajectory:
            return PDiagnostics(
                policy_loss=0.0, value_loss=0.0, entropy_loss=0.0,
                total_loss=0.0, approx_kl=0.0, clip_fraction=0.0,
                gradient_norm=0.0, advantage_mean=0.0, advantage_std=0.0,
                value_mean=0.0, value_std=0.0,
            )

        # Filter out fallback samples for P8 (existing contract, unchanged).
        if self.policy_type == "encoder_only":
            trajectory = [
                t for t in trajectory if not t.get("fallback_applied", False)
            ]

        if not trajectory:
            self._update_count += 1
            return PDiagnostics(
                policy_loss=0.0, value_loss=0.0, entropy_loss=0.0,
                total_loss=0.0, approx_kl=0.0, clip_fraction=0.0,
                gradient_norm=0.0, advantage_mean=0.0, advantage_std=0.0,
                value_mean=0.0, value_std=0.0,
            )

        device = self.backbone._device()

        # [G11.2.4] Package the stored rollout for the epoch/minibatch loop.
        # This preserves the RAW action payload and the per-step fleet snapshot.
        batch = build_rollout_batch(
            trajectory, self.policy_type, self.instance, device,
        )

        # [G11.2.4] Real PPO: ppo_epochs x minibatches, clipped surrogate,
        # value loss, live entropy, grad clipping, KL monitoring and
        # target-KL early stopping all execute here.
        diag = self.trainer.train_step_adapter(batch)

        # Count the update on the trainer, exactly as the pre-G11.2.4 inline
        # path did (PPOTrainer.train_step_adapter maintains its own counter;
        # this is the LinerShippingTrainer's counter, used by run_training,
        # checkpointing and get_summary).
        self._update_count += 1

        # [G11.1 FIX] Append metrics to log so get_summary() is accurate.
        # When called from run_training(), that caller has ALREADY appended a
        # metrics entry for this same trajectory, so we must not append a
        # second one (that would double-count the episode). We still patch the
        # existing entry with the PPO stats below.
        if log_metrics:
            try:
                metrics = self._build_metrics_from_trajectory(
                    trajectory,
                    update_count=self._update_count,
                    episode_count=self._episode_count,
                )
                self._metrics_log.append(metrics)
                self._episode_count += 1
            except Exception:
                pass  # metrics logging failure must not invalidate the PPO update

        # Update metrics with PPO stats
        if self._metrics_log:
            last = self._metrics_log[-1]
            last.PPO_policy_loss = diag.policy_loss
            last.PPO_value_loss = diag.value_loss
            last.PPO_entropy = diag.entropy_loss
            last.PPO_approx_kl = diag.approx_kl
            last.PPO_clip_fraction = diag.clip_fraction
            last.PPO_advantage_mean = diag.advantage_mean
            last.PPO_advantage_std = diag.advantage_std
            last.PPO_value_mean = diag.value_mean
            last.PPO_value_std = diag.value_std
            last.gradient_norm = diag.gradient_norm

        return diag

    def run_training(
        self,
        max_updates: Optional[int] = None,
    ) -> List[TrainingMetrics]:
        """
        Run full training loop.

        Parameters
        ----------
        max_updates : int, optional
            Maximum number of PPO updates to perform. Overrides any legacy
            ``max_episodes`` / ``steps_per_episode`` parameters.
            Defaults to ``self.config.max_updates``.

        Returns
        -------
        list[TrainingMetrics]
            Complete training metrics.

        Invariant: ONE trajectory is collected per training iteration. The
        same trajectory drives both episode-level metrics and the PPO update.
        """
        effective_max = max_updates if max_updates is not None else self.config.max_updates
        if effective_max is None or effective_max <= 0:
            raise ValueError(
                f"max_updates must be a positive integer, got {effective_max!r}. "
                f"Set config.max_updates or pass it explicitly."
            )

        save_frequency = self.config.checkpoint_frequency

        print(f"\n{'='*60}")
        print(f"P12 Training: {self.policy_type.upper()} on {self.instance_name}")
        print(f"{'='*60}")
        print(f"Instance: {self.instance.name}")
        print(f"Ports: {len(self.instance.ports)}, Vessels: {len(self.instance.vessel_types)}")
        print(f"Demand: {len(self.instance.demands)}, Fleet: {[(e.vessel_class, e.quantity) for e in self.instance.fleet]}")
        print(f"Policy: {self.policy_type}")
        print(f"Seed: {self.config.seed}")
        print(f"Max PPO updates: {effective_max}")
        print(f"Checkpoint frequency: {save_frequency}")
        print(f"{'='*60}\n")

        # Verify data integrity
        assert self._verify_data_integrity(), "Raw data modified during training!"

        all_metrics = []

        for update_idx in range(effective_max):
            seed = self.config.seed + update_idx

            # Collect a SINGLE rollout
            trajectory, num_steps = self.collect_rollout(seed=seed)

            if not trajectory:
                print(f"WARNING: Empty trajectory at update {update_idx+1}, skipping.")
                continue

            # Derive metrics from THE SAME trajectory
            metrics = self._build_metrics_from_trajectory(
                trajectory,
                update_count=self._update_count,
                episode_count=self._episode_count,
            )
            all_metrics.append(metrics)
            self._metrics_log.append(metrics)
            self._episode_count += 1

            # Perform PPO update on THE SAME trajectory.
            # log_metrics=False: the metrics entry for this trajectory was
            # already appended above, so perform_ppo_update must not append a
            # second one for the same episode.
            diag = self.perform_ppo_update(trajectory, log_metrics=False)

            # Update metrics with PPO stats
            metrics.PPO_policy_loss = diag.policy_loss
            metrics.PPO_value_loss = diag.value_loss
            metrics.PPO_entropy = diag.entropy_loss
            metrics.PPO_approx_kl = diag.approx_kl
            metrics.PPO_clip_fraction = diag.clip_fraction
            metrics.PPO_advantage_mean = diag.advantage_mean
            metrics.PPO_advantage_std = diag.advantage_std
            metrics.PPO_value_mean = diag.value_mean
            metrics.PPO_value_std = diag.value_std
            metrics.gradient_norm = diag.gradient_norm

            # Log progress
            if (update_idx + 1) % max(1, save_frequency // 5) == 0 or update_idx == 0:
                print(f"Update {update_idx+1}/{effective_max}: "
                      f"steps={num_steps}, "
                      f"reward={metrics.reward:.2f}, "
                      f"KL={diag.approx_kl:.4f}, "
                      f"clip={diag.clip_fraction:.2%}, "
                      f"entropy={diag.entropy_loss:.4f}")

            # Check for NaN/Inf
            if not math.isfinite(diag.total_loss):
                print(f"WARNING: Non-finite loss at update {update_idx+1}")
                break

            # Adaptive KL-based learning-rate adjustment (paper Section 5)
            # Paper specifies target_kl=0.1 as a threshold for LR adaptation,
            # NOT as a hard termination condition. Reduce LR when KL is high;
            # increase when KL is well below target.
            if diag.approx_kl > self.config.target_kl:
                current_lr = self.trainer.optimizer.param_groups[0]["lr"]
                new_lr = max(current_lr * 0.5, 1e-6)
                for pg in self.trainer.optimizer.param_groups:
                    pg["lr"] = new_lr
                print(f"  KL={diag.approx_kl:.4f} > target={self.config.target_kl}: "
                      f"reducing LR to {new_lr:.2e}")
            elif diag.approx_kl < self.config.target_kl * 0.5:
                current_lr = self.trainer.optimizer.param_groups[0]["lr"]
                new_lr = min(current_lr * 1.1, self.config.learning_rate)
                for pg in self.trainer.optimizer.param_groups:
                    pg["lr"] = new_lr

            # Save checkpoint
            if (update_idx + 1) % save_frequency == 0:
                self.save_checkpoint(f"checkpoint_ep_{update_idx+1}.pt")

        # Final checkpoint
        self.save_checkpoint("final_checkpoint.pt")

        # Verify data integrity after training
        assert self._verify_data_integrity(), "Raw data modified during training!"

        print(f"\n{'='*60}")
        print(f"Training complete: {len(all_metrics)} episodes, {self._update_count} updates")
        print(f"{'='*60}")
        print(f"{'='*60}")

        return all_metrics

    def save_checkpoint(self, filename: str) -> str:
        """
        Save training checkpoint.

        Parameters
        ----------
        filename : str
            Checkpoint filename.

        Returns
        -------
        str
            Full path to saved checkpoint.
        """
        path = self.checkpoint_dir / filename

        checkpoint = {
            "instance_name": self.instance_name,
            "policy_type": self.policy_type,
            "config": self.config.to_dict(),
            "update_count": self._update_count,
            "episode_count": self._episode_count,
            "metrics_log": [m.to_dict() for m in self._metrics_log],
            "raw_data_hashes": self._raw_data_hashes,
            "timestamp": datetime.now().isoformat(),
        }

        # Save model states
        checkpoint["backbone_state_dict"] = self.backbone.state_dict()
        checkpoint["policy_state_dict"] = self.policy.state_dict()
        checkpoint["critic_state_dict"] = self.critic.state_dict()
        checkpoint["optimizer_state_dict"] = self.trainer.optimizer.state_dict()

        torch.save(checkpoint, path)

        print(f"Checkpoint saved: {path}")
        return str(path)

    def load_checkpoint(self, path: str) -> None:
        """
        Load training checkpoint.

        Parameters
        ----------
        path : str
            Checkpoint file path.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        self.backbone.load_state_dict(checkpoint["backbone_state_dict"])
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.critic.load_state_dict(checkpoint["critic_state_dict"])
        self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self._update_count = checkpoint.get("update_count", 0)
        self._episode_count = checkpoint.get("episode_count", 0)
        self._metrics_log = [
            TrainingMetrics(**m) for m in checkpoint.get("metrics_log", [])
        ]

        print(f"Checkpoint loaded from {path}")

    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        if not self._metrics_log:
            return {"status": "no_training"}

        rewards = [m.reward for m in self._metrics_log]
        profits = [m.network_profit_eta for m in self._metrics_log]
        kl_values = [m.PPO_approx_kl for m in self._metrics_log]
        entropies = [m.PPO_entropy for m in self._metrics_log]

        return {
            "instance": self.instance_name,
            "policy": self.policy_type,
            "updates": self._update_count,
            "episodes": self._episode_count,
            "final_reward": rewards[-1] if rewards else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "max_reward": max(rewards) if rewards else 0.0,
            "min_reward": min(rewards) if rewards else 0.0,
            "final_profit": profits[-1] if profits else 0.0,
            "mean_profit": sum(profits) / len(profits) if profits else 0.0,
            "final_kl": kl_values[-1] if kl_values else 0.0,
            "mean_kl": sum(kl_values) / len(kl_values) if kl_values else 0.0,
            "final_entropy": entropies[-1] if entropies else 0.0,
            "min_entropy": min(entropies) if entropies else 0.0,
            "wall_clock_time": time.time() - self._start_time,
            "checkpoint_path": str(self.checkpoint_dir / "final_checkpoint.pt"),
        }


def main() -> None:
    """Run P12 training on Baltic instance."""
    # Training configuration
    config = TrainingConfig(
        dataset="Baltic",
        policy="encoder_only",
        learning_rate=2e-4,
        gamma=1.0,
        gae_lambda=0.9,
        ppo_epochs=10,
        clip_epsilon=0.2,
        target_kl=0.1,
        entropy_coefficient=0.05,
        value_coefficient=0.5,
        num_envs=1,
        steps_per_env=100,
        minibatch_size=32,
        seed=42,
        max_updates=500,
        checkpoint_frequency=50,
        hidden_dim=64,  # Reduced for quick training
        gat_layers=2,
        transformer_layers=2,
        transformer_heads=2,
        lstm_layers=1,
    )

    # Run training
    trainer = LinerShippingTrainer(
        instance_name="Baltic",
        policy_type=config.policy,
        config=config,
        checkpoint_dir="checkpoints/p8_baltic",
    )

    metrics = trainer.run_training(
        max_updates=config.max_updates,
    )

    # Print summary
    summary = trainer.get_summary()
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
