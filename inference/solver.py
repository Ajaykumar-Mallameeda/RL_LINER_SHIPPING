"""
P13 — Core Inference Solver.

Orchestrates the full inference loop: load checkpoint → build model →
reset environment → run policy until termination → evaluate final network.

Data flow:
    checkpoint (.pt)
        ↓ load_and_validate_checkpoint
    NeuralBackbone + Policy (+ Critic if needed)
        ↓
    LSNDPEnv (P4 environment)
        ↓ reset(seed)
    StateEncoder (P5) → GraphTensors
        ↓
    Policy inference (sample_action / deterministic_action)
        ↓
    ServiceAction → env.step()
        ↓
    MCF evaluation (P3) → η_t
        ↓
    InferenceResult serialization
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from data.instance import LINERLIBInstance
from data.linerlib_loader import LINERLIBLoader
from env.action import ServiceAction
from env.environment import LSNDPEnv, MAX_SERVICES_SAFETY_CAP
from mcf import evaluate_network
from mcf.expanded_graph import ServiceDefinition
from mcf.expanded_graph import ServiceDefinition
from mcf.result import MCFResult
from neural import ArchitectureConfig, NeuralBackbone, neural_state_to_tensors
from neural.tensors import GraphTensors
from policies.common import PolicyDiagnostics
from policies.encoder_decoder import EncoderDecoderPolicy, EncoderDecoderOutput
from policies.encoder_only import EncoderOnlyPolicy, EncoderOnlyOutput
from state.representation import ServiceMembership, StateEncoder

from .checkpoint import CheckpointError, load_and_validate_checkpoint
from .config import InferenceConfig
from .result import InferenceResult, ServiceStep

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class InferenceSolver:
    """
    Production-quality inference/solver for trained LSNDP RL policies.

    Parameters
    ----------
    checkpoint_path : str
        Path to a P10/P12 training checkpoint (.pt file).
    config : InferenceConfig
        Inference configuration controlling mode, validation, etc.
    dataset_root : str
        Root directory for LINERLIB data files (default: "data/").

    Usage
    -----
    >>> solver = InferenceSolver("checkpoints/final.pt", InferenceConfig())
    >>> result = solver.run(instance_name="Baltic", seed=42)
    >>> print(result.final_eta)
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[InferenceConfig] = None,
        dataset_root: str = "data",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.config = config or InferenceConfig()
        self.dataset_root = dataset_root

        # Loaded after __post_init().
        self._metadata: Optional[Any] = None
        self._backbone: Optional[NeuralBackbone] = None
        self._policy: Optional[Any] = None
        self._instance: Optional[LINERLIBInstance] = None
        self._dist_by_pair: Dict[Any, Any] = {}
        self._state_encoder: Optional[StateEncoder] = None
        self._generator: Optional[Any] = None
        self._checkpoint_payload: Optional[Dict[str, Any]] = None

        self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        """Load and validate the checkpoint, construct components."""
        payload, metadata = load_and_validate_checkpoint(
            self.checkpoint_path,
            expected_policy_type=self.config.policy_type,
        )
        self._metadata = metadata
        self._checkpoint_payload = payload

        # Load instance.
        loader = LINERLIBLoader(self.dataset_root)
        self._instance = loader.load(metadata.instance_name)
        self._dist_by_pair = {
            (a.origin, a.destination): a for a in self._instance.distances
        }

        # Reconstruct architecture config from checkpoint.
        arch_config = ArchitectureConfig(
            hidden_dim=metadata.backbone_config.get("hidden_dim", 512),
            gat_layers=metadata.backbone_config.get("gat_layers", 3),
            transformer_layers=metadata.backbone_config.get("transformer_layers", 3),
            transformer_heads=metadata.backbone_config.get("transformer_heads", 8),
            lstm_layers=metadata.backbone_config.get("lstm_layers", 1),
        )

        # Reconstruct backbone.
        self._backbone = NeuralBackbone(arch_config)
        self._backbone.load_state_dict(payload["backbone_state_dict"])
        self._backbone.eval()

        # Reconstruct policy.
        from actions.service_generator import ServiceGenerator
        self._generator = ServiceGenerator(self._instance, self._dist_by_pair)
        self._state_encoder = StateEncoder(self._instance, self._dist_by_pair)

        if self.config.policy_type == "encoder_only":
            self._policy = EncoderOnlyPolicy(
                self._backbone, self._instance, self._generator,
            )
        elif self.config.policy_type == "encoder_decoder":
            self._policy = EncoderDecoderPolicy(
                self._backbone, self._instance, self._generator,
            )
        else:
            raise CheckpointError(
                f"Unsupported policy_type: {self.config.policy_type}",
                details=["Must be 'encoder_only' or 'encoder_decoder'."],
            )

        self._policy.load_state_dict(payload["policy_state_dict"])
        self._policy.eval()

        logger.info(
            f"Components loaded: instance={metadata.instance_name}, "
            f"policy={self.config.policy_type}, "
            f"params={metadata.total_params:,}"
        )

    # ---- public API ----

    def run(
        self,
        instance_name: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> InferenceResult:
        """
        Run a single inference episode.

        Parameters
        ----------
        instance_name : str, optional
            Override the instance from the checkpoint. If None, uses the
            checkpoint's instance_name.
        seed : int, optional
            Override the seed. If None, uses config.seed.

        Returns
        -------
        InferenceResult
            Complete result of the inference episode.
        """
        start_time = time.time()

        # Reconstruct instance if needed.
        checkpoint_inst = getattr(self._metadata, "instance_name", "Baltic")
        inst_name = instance_name or checkpoint_inst
        effective_seed = seed if seed is not None else (self.config.seed or 0)

        # Track whether we're using the original training instance.
        using_different_instance = (inst_name != checkpoint_inst)

        # Create environment with the (possibly overridden) instance.
        if using_different_instance:
            loader = LINERLIBLoader(self.dataset_root)
            self._instance = loader.load(inst_name)
            self._dist_by_pair = {
                (a.origin, a.destination): a for a in self._instance.distances
            }
            self._state_encoder = StateEncoder(self._instance, self._dist_by_pair)
            # Note: backbone/policy were built for the checkpoint instance;
            # using a different instance is allowed but the model was trained
            # on different data — this is intentional for generalization tests.

        steps: List[ServiceStep] = []
        warnings: List[str] = []
        errors: List[str] = []
        diagnostics_data: Optional[Dict[str, Any]] = None

        # Guard: if using a different instance, check port compatibility.
        if using_different_instance:
            n_ports_env = len(self._instance.ports)
            try:
                n_ports_policy = len(self._policy.port_codes)
            except AttributeError:
                n_ports_policy = 0
            if n_ports_env != n_ports_policy and n_ports_policy > 0:
                warnings.append(
                    f"Cross-instance inference: model trained on {checkpoint_inst} "
                    f"({n_ports_policy} ports), running on {inst_name} "
                    f"({n_ports_env} ports). Results may not generalize."
                )

        env = LSNDPEnv(self._instance)
        obs, info = env.reset(seed=effective_seed)

        # Track vessel requirements alongside services for proper state encoding.
        vessel_requirements: Dict[str, Dict[str, float]] = {}

        # Encode initial state.
        rem_demand, fleet_remaining, membership = self._extract_env_state(
            obs, vessel_requirements,
        )
        bundle = self._encode_state(rem_demand, fleet_remaining, membership)

        step_idx = 0
        while not env.is_terminal() and step_idx < self.config.max_services:
            try:
                step_result, new_obs = self._inference_step(
                    env, obs, rem_demand, fleet_remaining,
                    membership, vessel_requirements, bundle,
                    step_idx, effective_seed,
                )
                if step_result is None:
                    # Action rejected — skip step but don't increment counter.
                    continue
                steps.append(step_result)
                step_idx += 1

                # Update state for next iteration.
                obs = new_obs
                rem_demand, fleet_remaining, membership = self._extract_env_state(
                    obs, vessel_requirements,
                )
                bundle = self._encode_state(rem_demand, fleet_remaining, membership)

            except ValueError as e:
                # Port count mismatch or other structural incompatibility.
                # This is expected when running on a different instance than
                # the training data. Record as a non-fatal warning.
                errors.append(f"Step {step_idx}: structural incompatibility: {e}")
                logger.warning(f"Inference step {step_idx} structural error: {e}")
                break
            except Exception as e:  # noqa: BLE001
                errors.append(f"Step {step_idx} failed: {type(e).__name__}: {e}")
                logger.warning(f"Inference step {step_idx} error: {e}")
                break

        # Final MCF evaluation.
        final_mcf = self._final_mcf_evaluation(env)

        runtime = time.time() - start_time

        # Determine termination reason.
        term_reason = env.get_state().termination_reason
        is_truncated = env.is_terminal() and term_reason == "safety_cap_reached"
        if term_reason is None and env.is_terminal():
            term_reason = "unknown"

        # Record diagnostics if requested.
        if self.config.record_diagnostics:
            diagnostics_data = {
                "num_steps": len(steps),
                "runtime_seconds": runtime,
                "instance_name": self._instance.name,
                "policy_type": self.config.policy_type,
            }

        result = InferenceResult(
            dataset=self._instance.name,
            instance_name=self._instance.name,
            policy_type=self.config.policy_type,
            checkpoint_path=self.checkpoint_path,
            checkpoint_hash=self._metadata.checkpoint_hash,  # type: ignore
            seed=effective_seed,
            deterministic=self.config.deterministic,
            services=steps,
            total_services=len(steps),
            final_eta=final_mcf.eta if final_mcf else 0.0,
            final_mcf_result=final_mcf.summary() if final_mcf else None,
            runtime_seconds=runtime,
            termination_reason=term_reason,
            is_truncated=is_truncated,
            warnings=warnings,
            errors=errors,
            diagnostics=diagnostics_data,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            software_version={
                "torch": torch.__version__,
                "numpy": __import__("numpy").__version__,
            },
        )

        logger.info(
            f"Inference complete: eta={result.final_eta:,.2f}, "
            f"services={result.total_services}, "
            f"term={result.termination_reason}, trunc={result.is_truncated}, "
            f"runtime={runtime:.2f}s"
        )

        return result

    def run_multiple(
        self,
        n_runs: int,
        instance_name: Optional[str] = None,
        base_seed: Optional[int] = None,
    ) -> List[InferenceResult]:
        """
        Run multiple inference episodes (for stochastic evaluation).

        Parameters
        ----------
        n_runs : int
            Number of independent runs.
        instance_name : str, optional
            Instance to evaluate on.
        base_seed : int, optional
            Base seed; each run uses base_seed + i.

        Returns
        -------
        list[InferenceResult]
            One result per run.
        """
        seeds = [
            (base_seed or self.config.seed or 0) + i
            for i in range(n_runs)
        ]
        return [
            self.run(instance_name=instance_name, seed=s)
            for s in seeds
        ]

    # ---- internal helpers ----

    def _extract_env_state(
        self,
        obs: Dict[str, Any],
        vessel_requirements: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[Dict[int, float], Dict[str, float], ServiceMembership]:
        """Extract P5-compatible state components from P4 observation."""
        n_demands = len(obs["remaining_demand"])
        rem_demand = {
            i: float(obs["remaining_demand"][i])
            for i in range(n_demands)
        }

        fleet: Dict[str, float] = {}
        vessel_classes = sorted(self._instance.vessel_types.keys())
        for i, vc in enumerate(vessel_classes):
            fleet[vc] = float(obs["fleet_remaining"][i])

        # Build service membership from env observation.
        # Services are stored in obs["services"] as dicts.
        membership = ServiceMembership()
        services_list = obs.get("services", [])
        for svc_obs in services_list:
            svc_def = ServiceDefinition(
                service_id=len(membership.service_defs),
                vessel_class=svc_obs["vessel_class"],
                port_sequence=list(svc_obs["port_sequence"]),
            )
            # Look up vessel requirements from env's internal tracking.
            n_vs_vals: Dict[str, float] = {}
            if vessel_requirements is not None:
                sid_str = str(svc_def.service_id)
                n_vs_vals = dict(vessel_requirements.get(sid_str, {}))
            membership.add(svc_def, n_vs_vals)

        return rem_demand, fleet, membership

    def _encode_state(
        self,
        rem_demand: Dict[int, float],
        fleet_remaining: Dict[str, float],
        membership: ServiceMembership,
    ) -> GraphTensors:
        """Encode current state to neural tensor bundle."""
        ns = self._state_encoder.encode(rem_demand, fleet_remaining, membership)
        return neural_state_to_tensors(ns)

    def _inference_step(
        self,
        env: LSNDPEnv,
        obs: Dict[str, Any],
        rem_demand: Dict[int, float],
        fleet_remaining: Dict[str, float],
        membership: ServiceMembership,
        vessel_requirements: Dict[str, Dict[str, float]],
        bundle: GraphTensors,
        step_idx: int,
        seed: int,
    ) -> Tuple[Optional[ServiceStep], Dict[str, Any]]:
        """
        Execute one inference step: policy forward → action → env step.

        Returns
        -------
        tuple[ServiceStep or None, dict]
            (step_result, new_observation). step_result is None if action was
            rejected; new_observation is the updated P4 observation.
        """
        # Run policy inference.
        if self.config.policy_type == "encoder_only":
            out: Any = self._run_encoder_only(bundle, fleet_remaining, seed)
        else:
            out = self._run_encoder_decoder(bundle, fleet_remaining, seed)

        # Extract service action.
        sa = self._extract_service_action(out)
        if sa is None:
            return None, obs

        # Validate action before executing.
        if self.config.validate_actions:
            validation_reasons = env._validate_service(sa)  # noqa: SLF001
            if validation_reasons:
                logger.warning(
                    f"Step {step_idx}: action rejected by env validation: "
                    f"{validation_reasons}"
                )
                return None, obs

        # Execute in environment.
        try:
            obs_out, reward_norm, terminated, truncated, info = env.step(sa)
        except Exception as e:
            logger.warning(f"Step {step_idx}: env.step failed: {e}")
            return None, obs

        # Extract vessel requirement from env state for membership tracking.
        env_state = env.get_state()
        sid_key = str(env_state.num_services_added - 1)
        vr = env_state.vessel_requirements.get(sid_key, {})
        vessel_requirements[sid_key] = vr

        # Record step.
        eta = info.get("eta", 0.0)
        reward_raw = info.get("reward_raw", 0.0)

        # Re-extract fleet after step.
        new_rem, new_fleet, new_membership = self._extract_env_state(
            obs_out, vessel_requirements,
        )

        step = ServiceStep(
            step_index=step_idx,
            vessel_class=sa.vessel_class,
            port_sequence=list(sa.port_sequence),
            service_id=sa.service_id or step_idx,
            log_prob=self._get_log_prob(out),
            entropy=self._get_entropy(out),
            reward_raw=reward_raw,
            reward_normalized=reward_norm,
            eta_cumulative=eta,
            fleet_after=dict(new_fleet),
            terminated=terminated,
            truncated=truncated,
            termination_reason=info.get("termination_reason"),
        )

        return step, obs_out

    def _run_encoder_only(
        self,
        bundle: GraphTensors,
        fleet_remaining: Dict[str, float],
        seed: int,
    ) -> EncoderOnlyOutput:
        """Run encoder-only policy inference."""
        policy = self._policy  # type: ignore — set in __init__
        assert isinstance(policy, EncoderOnlyPolicy)
        if self.config.deterministic:
            return policy.deterministic_action(bundle, fleet_remaining)
        else:
            return policy.sample_action(bundle, fleet_remaining, seed=seed)

    def _run_encoder_decoder(
        self,
        bundle: GraphTensors,
        fleet_remaining: Dict[str, float],
        seed: int,
    ) -> EncoderDecoderOutput:
        """Run encoder-decoder policy inference."""
        policy = self._policy  # type: ignore
        assert isinstance(policy, EncoderDecoderPolicy)
        if self.config.deterministic:
            return policy.deterministic_action(bundle, fleet_remaining)
        else:
            return policy.sample_action(bundle, fleet_remaining, seed=seed)

    def _extract_service_action(
        self, output: Any,
    ) -> Optional[ServiceAction]:
        """
        Extract a ServiceAction from policy output.

        Returns None if no valid action was produced.
        """
        if self.config.policy_type == "encoder_only":
            assert isinstance(output, EncoderOnlyOutput)
            return output.service_action
        else:
            assert isinstance(output, EncoderDecoderOutput)
            return output.service_action

    def _get_log_prob(self, output: Any) -> Optional[float]:
        """Extract log-probability from policy output."""
        if self.config.policy_type == "encoder_only":
            assert isinstance(output, EncoderOnlyOutput)
            lp = output.raw_log_prob
            return float(lp.item()) if lp.numel() == 1 else None
        else:
            assert isinstance(output, EncoderDecoderOutput)
            lp = output.log_prob
            return float(lp.item()) if lp is not None and lp.numel() == 1 else None

    def _get_entropy(self, output: Any) -> Optional[float]:
        """Extract entropy from policy output."""
        if self.config.policy_type == "encoder_only":
            assert isinstance(output, EncoderOnlyOutput)
            ent = output.entropy
            return float(ent.item()) if ent.numel() == 1 else None
        else:
            assert isinstance(output, EncoderDecoderOutput)
            ent = output.entropy
            return float(ent.item()) if ent is not None and ent.numel() == 1 else None

    def _final_mcf_evaluation(
        self, env: LSNDPEnv,
    ) -> Optional[MCFResult]:
        """
        Run final MCF evaluation on the completed network.

        Uses the environment's internal state which already contains all
        services and vessel requirements.
        """
        state = env.get_state()
        if not state.services:
            # No services were added — return empty MCF result.
            return MCFResult(
                instance_name=self._instance.name,
                num_services=0,
                num_demands=len(self._instance.demands),
            )

        result = evaluate_network(
            instance=self._instance,
            services=state.services,
            vessel_requirements=state.vessel_requirements,
        )
        return result

    @property
    def metadata(self) -> Optional[CheckpointMetadata]:
        """Return checkpoint metadata, or None if not yet loaded."""
        return self._metadata

    @property
    def instance(self) -> Optional[LINERLIBInstance]:
        """Return the currently loaded instance."""
        return self._instance
