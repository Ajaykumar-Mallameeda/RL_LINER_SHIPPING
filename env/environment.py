"""
P4 — Liner Shipping Network Design Problem (LSNDP) Gymnasium Environment.

Implements a paper-faithful RL environment for network design. The agent builds
a liner shipping network one service at a time; each accepted service triggers
a full MCF re-evaluation (per paper assumption M5) and produces an incremental
reward signal.

Architecture:
  LINERLIB Instance → LSNDPEnv.step(action) → Validate → Add Service →
  P3.evaluate_network() → η_t → reward → state transition

Source-of-truth hierarchy:
  1. docs/PAPER_METHOD_SPECIFICATION.md (paper algorithm, reward, termination)
  2. docs/PROBLEM_FORMULATION.md (P2 mathematical contract)
  3. data.instance (P1 canonical types)
  4. mcf.evaluate_network (P3 evaluation authority)

Evidence tags throughout document paper vs implementation decisions.

Prohibited: no GAT, Transformer, LSTM, PPO, training loops, learned policies,
            neural embeddings, contribution-margin rewards, transit-time optimization,
            or reward shaping beyond the paper's normalized incremental reward.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

from data.instance import FleetEntry, LINERLIBInstance, Port, VesselType
from mcf import ServiceDefinition, evaluate_network
from mcf.result import MCFResult

from .action import ServiceAction


# ---------------------------------------------------------------------------
# Constants and defaults
# ---------------------------------------------------------------------------

# Engineering safety cap: maximum services per episode.
# The paper does not specify a hard |S|_max in Algorithm 2.
# This cap prevents infinite episodes in pathological cases but is NOT a
# paper termination condition. Episodes hitting this are truncated=True.
MAX_SERVICES_SAFETY_CAP = 100

# Floating-point tolerance for profit comparisons.
_PROFIT_TOL = 1e-6


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

class ServiceValidationError(Exception):
    """Raised when a service action fails structural validation."""

    def __init__(self, service_action: ServiceAction, reasons: List[str]):
        self.service_action = service_action
        self.reasons = reasons
        msg = "Service action validation failed:\n" + "\n".join(f"  - {r}" for r in reasons)
        super().__init__(msg)


# ---------------------------------------------------------------------------
# Internal state dataclass
# ---------------------------------------------------------------------------

@dataclass
class _EnvState:
    """
    Internal mutable state for a single episode.

    Parameters
    ----------
    instance :
        The loaded LINERLIBInstance (read-only reference).
    services :
        List of ServiceDefinition added so far (ordered by insertion).
    vessel_requirements :
        vessel_requirements[service_id_str][vessel_class] = n_vs.
    remaining_demand :
        Remaining unsatisfied demand per commodity index.
        initialized from instance.demands.
    fleet_remaining :
        Remaining vessel count per class as float (fractional consumption via n_vs).
        Initialized from instance.fleet; decremented by n_vs each step.
    profit_history :
        [η_0, η_1, ..., η_t] where η_0 = 0 (empty network), η_1 after first service.
    num_services_added :
        Total number of services successfully added.
    demand_satisfied_total :
        Cumulative FFE/week satisfied across all demands (from latest MCF eval).
    demand_rejected_total :
        Cumulative FFE/week rejected across all demands (from latest MCF eval).
    total_demand :
        Sum of all d_q (constant across episode).
    last_mcf_result :
        Most recent MCFResult from evaluate_network.
    seed :
        Random seed used for this episode (for determinism verification).
    termination_reason :
        Last reason the episode ended (None if still active).
    """
    instance: LINERLIBInstance
    services: List[ServiceDefinition] = field(default_factory=list)
    vessel_requirements: Dict[str, Dict[str, float]] = field(
        default_factory=dict,
    )
    remaining_demand: Dict[int, float] = field(default_factory=dict)
    fleet_remaining: Dict[str, float] = field(default_factory=dict)
    profit_history: List[float] = field(default_factory=list)
    num_services_added: int = 0
    demand_satisfied_total: float = 0.0
    demand_rejected_total: float = 0.0
    total_demand: float = 0.0
    last_mcf_result: Optional[MCFResult] = None
    seed: Optional[int] = None
    termination_reason: Optional[str] = None

    def __post_init__(self) -> None:
        """Initialize demand and fleet state from instance."""
        for idx, dem in enumerate(self.instance.demands):
            self.remaining_demand[idx] = dem.ffe_per_week
        self.total_demand = sum(d.ffe_per_week for d in self.instance.demands)
        for entry in self.instance.fleet:
            self.fleet_remaining[entry.vessel_class] = float(entry.quantity)
        # η_0 = 0 for empty network [PAPER — CONFIRMED]
        self.profit_history.append(0.0)


# ---------------------------------------------------------------------------
# Observation space definitions
# ---------------------------------------------------------------------------

def _make_observation_space(instance: LINERLIBInstance) -> gym.spaces.Space:
    """
    Construct the Gymnasium observation space for P4.

    P4 returns raw (non-tensor) state — a serializable dict.
    P5 will later define the GAT-ready tensor representation.

    We track key state via fixed-shape arrays; services list is kept
    internally and accessible via get_state().
    """
    n_demands = len(instance.demands)
    n_vessel_classes = len(instance.vessel_types)
    total_vessels = sum(e.quantity for e in instance.fleet)

    return gym.spaces.Dict({
        "remaining_demand": gym.spaces.Box(
            low=0.0,
            high=np.inf,
            shape=(n_demands,),
            dtype=np.float64,
        ),
        "fleet_remaining": gym.spaces.Box(
            low=0,
            high=max(total_vessels, 1),
            shape=(n_vessel_classes,),
            dtype=np.int64,
        ),
        "service_count": gym.spaces.Discrete(MAX_SERVICES_SAFETY_CAP + 1),
        "last_profit": gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(1,),
            dtype=np.float64,
        ),
        "instance_info": gym.spaces.Dict({
            "num_ports": gym.spaces.Discrete(len(instance.ports) + 1),
            "num_vessel_classes": gym.spaces.Discrete(n_vessel_classes + 1),
            "total_vessels": gym.spaces.Discrete(total_vessels + 1),
            "num_demands": gym.spaces.Discrete(n_demands + 1),
            "name": gym.spaces.Text(max_length=64),
        }),
    })


def _make_action_space(instance: LINERLIBInstance) -> gym.spaces.Space:
    """
    Construct the Gymnasium action space.

    [ENGINEERING DECISION]: We define a composite action space that allows
    the agent to choose both a vessel class and a port sequence. In practice,
    P6 will generate structured ServiceAction objects; the env accepts
    either the object directly or a dict representation.

    Action structure:
      - vessel_class_idx: discrete index into instance.vessel_types
      - port_sequence: Sequence of Discrete(port_count) — variable length
    """
    vessel_classes = sorted(instance.vessel_types.keys())
    ports_sorted = sorted(instance.ports.keys())

    return gym.spaces.Dict({
        "vessel_class": gym.spaces.Discrete(len(vessel_classes)),
        # Variable-length sequence of port indices.
        "port_sequence": gym.spaces.Sequence(
            gym.spaces.Discrete(len(ports_sorted)),
        ),
    })


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class LSNDPEnv(gym.Env):
    """
    Gymnasium environment for the Liner Shipping Network Design Problem.

    The environment represents a single MDP episode in which the agent
    incrementally builds a liner shipping network by adding services.

    Paper reference: Dutta et al. (2024), arXiv:2411.09068.
    - Action per step: add one complete round-trip service.
    - Transition: evaluate network through greedy MCF, update profit.
    - Reward: R_{t+1} = (η_{t+1} - η_t) / η_1  [PAPER Eq. 36].
    - Termination: vessel exhaustion or demand satisfaction [PAPER Alg. 2].

    Fleet semantics resolution (OQ-1):
      The paper treats the fleet constraint as SOFT — deviations are handled
      economically via C_unused (Eq. 34), not as hard feasibility constraints.
      Therefore:
        - Actions are NOT masked based on remaining fleet.
        - A service requiring more vessels than available can still be added.
        - Over-utilization incurs cost; under-utilization generates profit.
        - "Vessel exhaustion" termination checks remaining fleet counts == 0,
          but note that fractional vessel requirements mean exhaustion is
          rare and may never naturally occur.

    Demand-state handling:
      After each service addition, the MCF is re-run on the complete network.
      The resulting routed/rejected demand updates the remaining_demand state.
      Demand satisfaction termination occurs when all demands are fully routed.

    Termination / truncation:
      - terminated=True: natural episode end (vessel exhaustion or demand sat).
      - truncated=True: engineering safety cap reached (|S| > MAX_SERVICES_SAFETY_CAP).
    """

    metadata = {"render_modes": None}

    def __init__(
        self,
        instance: LINERLIBInstance,
        render_mode: Optional[str] = None,
    ) -> None:
        """
        Initialize the LSNDP environment.

        Parameters
        ----------
        instance :
            A loaded LINERLIBInstance (P1 data foundation). Read-only reference.
        render_mode :
            Not used — logging/rendering deferred to P5+.
        """
        super().__init__()

        self._instance = instance
        self._vessel_classes = sorted(instance.vessel_types.keys())
        self._ports_sorted = sorted(instance.ports.keys())
        self._obs_space = _make_observation_space(instance)
        self._act_space = _make_action_space(instance)
        self.resetted = False

        # Build distance lookup: (origin, dest) -> DistanceArc
        self._dist_by_pair: Dict[Tuple[str, str], Any] = {}
        for arc in instance.distances:
            self._dist_by_pair[(arc.origin, arc.destination)] = arc

    # ---- Gymnasium API ----

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._obs_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._act_space

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Reset the environment to initial state (empty network).

        Returns
        -------
        observation : dict
            Initial observation with empty services, full demand, full fleet.
        info : dict
            Initial info dictionary.
        """
        super().reset(seed=seed)
        self.resetted = True

        # Re-create state from scratch to ensure determinism.
        state = _EnvState(instance=self._instance, seed=seed)

        # Compute initial η_0 = 0 for empty network.
        # [PAPER CONFIRMED] Empty network has zero profit.
        assert len(state.profit_history) == 1
        assert math.isclose(state.profit_history[0], 0.0, abs_tol=_PROFIT_TOL)

        self._state = state
        self._step_count = 0
        self._terminated = False
        self._truncated = False

        obs = self._build_observation()
        info = self._build_info()
        return obs, info

    def step(
        self,
        action,
    ) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute one environment step: validate action, add service, evaluate.

        Parameters
        ----------
        action :
            Either a ``ServiceAction`` object or a dict matching the action space
            with keys ``vessel_class`` (int index) and ``port_sequence`` (list of int indices).

        Returns
        -------
        observation : dict
            Updated observation after service addition.
        reward : float
            Normalized incremental reward R_{t+1} = (η_{t+1} - η_t) / η_1.
        terminated : bool
            Whether a natural termination condition was met.
        truncated : bool
            Whether an engineering safety truncation occurred.
        info : dict
            Diagnostic information including profit, reward components, etc.
        """
        if self._terminated or self._truncated:
            raise RuntimeError(
                "Environment is in terminal state. Call reset() before stepping again."
            )

        # ---- 1. Convert action to ServiceAction ----
        sa = self._parse_action(action)

        # ---- 2. Validate service ----
        reasons = self._validate_service(sa)
        if reasons:
            raise ServiceValidationError(sa, reasons)

        # ---- 3. Build ServiceDefinition and compute vessel requirement ----
        vclass = sa.vessel_class
        vt = self._instance.vessel_types[vclass]
        sid = self._state.num_services_added
        svc_def = ServiceDefinition(
            service_id=sid,
            vessel_class=vclass,
            port_sequence=sa.port_sequence,
        )

        # Compute n_{v,s} = L_s / (v_s * 24 * 7) [PAPER App. A.3]
        # design_speed is in KNOTS (nm/hour) per fleet_data.csv spec,
        # so we convert to nm/week: knots * 24 hrs/day * 7 days/week.
        # Paper formula: n_vs = tour_dist / (speed * 7) assumes speed in nm/day.
        tour_dist = self._compute_tour_distance(svc_def)
        n_vs = tour_dist / (vt.design_speed * 24.0 * 7.0)

        self._state.services.append(svc_def)
        self._state.vessel_requirements[sid] = {vclass: n_vs}
        self._state.num_services_added += 1

        # ---- 3b. Update remaining fleet (fractional consumption) ----
        # Per paper Eq. 5/Appendix A.3: each service consumes n_vs vessels
        # of its assigned class from the fleet.
        self._state.fleet_remaining[vclass] = max(
            0.0, self._state.fleet_remaining.get(vclass, 0.0) - n_vs
        )

        # ---- 4. Evaluate network through P3 MCF ----
        result = evaluate_network(
            instance=self._instance,
            services=self._state.services,
            vessel_requirements=self._state.vessel_requirements,
        )
        self._state.last_mcf_result = result

        # ---- 5. Update profit history ----
        eta_t = result.eta
        self._state.profit_history.append(eta_t)

        # ---- 6. Update demand state from MCF result ----
        self._update_demand_state(result)

        # ---- 7. Calculate reward ----
        reward_raw, reward_normalized = self._compute_reward(result)

        # ---- 8. Check termination ----
        terminated = self._check_termination()
        truncated = False

        if not terminated and self._state.num_services_added > MAX_SERVICES_SAFETY_CAP:
            truncated = True
            self._state.termination_reason = "safety_cap_reached"

        if terminated or truncated:
            self._terminated = terminated
            self._truncated = truncated

        self._step_count += 1

        obs = self._build_observation()
        info = self._build_info(reward_raw, reward_normalized)
        return obs, reward_normalized, terminated, truncated, info

    # ---- Internal methods ----

    def _parse_action(self, action) -> ServiceAction:
        """Convert various action formats into a ServiceAction."""
        if isinstance(action, ServiceAction):
            return action

        if isinstance(action, dict):
            vclass_idx = int(action["vessel_class"])
            port_indices = list(action["port_sequence"])
            vclass = self._vessel_classes[vclass_idx]
            ports = [self._ports_sorted[i] for i in port_indices]
            return ServiceAction(vessel_class=vclass, port_sequence=ports)

        raise ValueError(
            f"Action must be a ServiceAction or dict, got {type(action).__name__}"
        )

    def _validate_service(
        self, sa: ServiceAction,
    ) -> List[str]:
        """
        Validate structural feasibility of a service action.

        Checks:
          1. Vessel class exists in instance.
          2. All ports exist in instance.
          3. At least 2 ports (minimum meaningful cycle).
          4. No duplicate ports in sequence (except implicit cycle closure).
          5. Distance exists for each consecutive port pair.
          6. Draft compatibility: vessel draft >= port draft for all ports.
          7. Service can be represented by P3 evaluator.

        Returns list of failure reasons (empty = valid).
        """
        reasons: List[str] = []

        # Check 1: Vessel class exists
        if sa.vessel_class not in self._instance.vessel_types:
            reasons.append(
                f"Vessel class '{sa.vessel_class}' not in instance vessel_types."
            )

        # Check 2 & 3: Ports exist and minimum length
        if not sa.port_sequence:
            reasons.append("Port sequence is empty.")
        else:
            for p in sa.port_sequence:
                if p not in self._instance.ports:
                    reasons.append(f"Port '{p}' not in instance ports.")
            if len(sa.port_sequence) < 2:
                reasons.append(
                    "Port sequence must contain at least 2 ports for a cycle."
                )

        # Check 4: No duplicate ports within the sequence
        if len(sa.port_sequence) != len(set(sa.port_sequence)):
            reasons.append(
                "Duplicate ports in sequence (each port should appear once, "
                "cycle closes implicitly)."
            )

        # Check 5: Distance existence for consecutive port pairs.
        # [PAPER] Draft is NOT a hard constraint — C_unused handles fleet
        # deviations economically per Eq. 34. We validate distance only here.
        if sa.vessel_class in self._instance.vessel_types and sa.port_sequence:
            vt = self._instance.vessel_types[sa.vessel_class]
            n_ports = len(sa.port_sequence)
            for i in range(n_ports):
                p_from = sa.port_sequence[i]
                p_to = sa.port_sequence[(i + 1) % n_ports]

                arc = self._dist_by_pair.get((p_from, p_to))
                if arc is None or arc.distance_nm <= 0:
                    reasons.append(
                        f"No valid distance for leg {p_from}→{p_to}."
                    )

        return reasons

    def _compute_tour_distance(self, svc: ServiceDefinition) -> float:
        """Compute total tour distance L_s for a service."""
        seq = svc.port_sequence
        total = 0.0
        n = len(seq)
        for i in range(n):
            arc = self._dist_by_pair.get((seq[i], seq[(i + 1) % n]))
            if arc is not None:
                total += arc.distance_nm
        return total

    def _update_demand_state(self, result: MCFResult) -> None:
        """
        Update remaining demand state from MCF evaluation result.

        For each commodity, subtract satisfied flow from remaining demand.
        Demands that are fully satisfied are set to 0.
        """
        for cr in result.commodity_results:
            idx = cr.commodity_idx
            self._state.remaining_demand[idx] = max(
                0.0,
                self._state.remaining_demand.get(idx, 0.0) - cr.satisfied,
            )

        self._state.demand_satisfied_total = result.routed_demand
        self._state.demand_rejected_total = result.rejected_demand

    def _compute_reward(
        self, result: MCFResult,
    ) -> Tuple[float, float]:
        """
        Compute raw and normalized incremental reward.

        Raw: R^{raw}_{t+1} = η_{t+1} - η_t
        Normalized: R_{t+1} = (η_{t+1} - η_t) / η_1

        [PAPER] Eq. 1, Eq. 36.
        """
        ph = self._state.profit_history
        eta_t_plus_1 = result.eta
        eta_t = ph[-2] if len(ph) >= 2 else 0.0
        raw = eta_t_plus_1 - eta_t

        # Normalization by η_1
        if len(ph) >= 2:
            eta_1 = ph[1]  # profit after first service
        else:
            eta_1 = 1.0  # fallback to avoid division by zero on first step

        # Edge case: if η_1 is zero, the paper doesn't specify behavior.
        # We use raw reward as a fallback (engineering decision).
        if math.isclose(eta_1, 0.0, abs_tol=_PROFIT_TOL):
            normalized = raw
        else:
            normalized = raw / eta_1

        return raw, normalized

    def _check_termination(self) -> bool:
        """
        Check paper-specified termination conditions.

        Condition A — vessel exhaustion [PAPER Alg. 2]:
          v_n <= 0 for all vessel classes.
          NOTE: Since vessel requirements are fractional and the paper treats
          fleet as soft, "exhaustion" here means remaining integer fleet is 0.

        Condition B — demand satisfaction [PAPER Alg. 2]:
          All demands fully satisfied (Dm_d = 0 for all d).

        Note: The paper also mentions a maximum service bound |S|_max but does
        not enforce it in Algorithm 2. We use an engineering safety cap instead
        (truncated, not terminated).

        Returns
        -------
        bool: True if episode should terminate naturally.
        """
        # Condition A: vessel exhaustion
        # Check if any vessel class still has remaining fleet.
        # Under the soft fleet interpretation, we only consider natural
        # exhaustion (remaining count == 0 for ALL classes).
        any_vessel_remaining = any(
            qty > 0 for qty in self._state.fleet_remaining.values()
        )
        if not any_vessel_remaining:
            self._state.termination_reason = "vessel_exhaustion"
            return True

        # Condition B: demand satisfaction
        # All remaining demands must be zero (or near-zero).
        total_remaining = sum(
            d for d in self._state.remaining_demand.values()
        )
        if total_remaining <= 1e-6:
            self._state.termination_reason = "demand_satisfied"
            return True

        return False

    def _build_observation(self) -> Dict:
        """Build the P4 observation dict (raw state, not neural tensors)."""
        # Services (kept as list for inspection, not part of gym space)
        services_list = []
        for svc in self._state.services:
            services_list.append({
                "vessel_class": svc.vessel_class,
                "port_sequence": list(svc.port_sequence),
            })

        # Remaining demand as numpy array
        n_demands = len(self._state.instance.demands)
        rem_demand = np.zeros(n_demands, dtype=np.float64)
        for idx, val in self._state.remaining_demand.items():
            rem_demand[idx] = val

        # Fleet remaining as numpy array
        n_vc = len(self._state.instance.vessel_types)
        fleet_arr = np.zeros(n_vc, dtype=np.int64)
        for i, vc in enumerate(self._vessel_classes):
            fleet_arr[i] = self._state.fleet_remaining.get(vc, 0)

        # Current profit
        current_profit = np.array(
            [self._state.profit_history[-1]], dtype=np.float64
        ) if self._state.profit_history else np.array([0.0], dtype=np.float64)

        # Instance info
        instance_info = {
            "num_ports": len(self._state.instance.ports),
            "num_vessel_classes": n_vc,
            "total_vessels": sum(e.quantity for e in self._state.instance.fleet),
            "num_demands": n_demands,
            "name": self._state.instance.name,
        }

        return {
            "remaining_demand": rem_demand,
            "fleet_remaining": fleet_arr,
            "service_count": self._state.num_services_added,
            "last_profit": current_profit,
            "instance_info": instance_info,
        }

    def _build_info(
        self,
        reward_raw: Optional[float] = None,
        reward_normalized: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build the info diagnostic dictionary."""
        result = self._state.last_mcf_result

        info: Dict[str, Any] = {
            "profit": self._state.profit_history[-1] if self._state.profit_history else 0.0,
            "profit_history": list(self._state.profit_history),
            "num_services": self._state.num_services_added,
            "step": self._step_count,
            "remaining_demand": dict(self._state.remaining_demand),
            "demand_satisfied": self._state.demand_satisfied_total,
            "demand_rejected": self._state.demand_rejected_total,
            "total_demand": self._state.total_demand,
            "vessel_state": dict(self._state.fleet_remaining),
            "vessel_requirements": {
                k: dict(v) for k, v in self._state.vessel_requirements.items()
            },
            "termination_reason": self._state.termination_reason,
        }

        if result is not None:
            info.update({
                "reward_raw": reward_raw if reward_raw is not None else 0.0,
                "reward_normalized": reward_normalized if reward_normalized is not None else 0.0,
                "demand_coverage": result.demand_coverage,
                "eta": result.eta,
                "total_revenue": result.total_revenue,
                "rejection_cost": result.rejection_cost,
                "handling_cost": result.handling_cost,
                "service_cost": result.service_cost,
                "unused_vessel_cost": result.unused_vessel_cost,
                "voyage_cost": result.voyage_cost,
                "port_call_cost": result.port_call_cost,
                "sailing_fuel_cost": result.sailing_fuel_cost,
                "idle_fuel_cost": result.idle_fuel_cost,
                "canal_fee_cost": result.canal_fee_cost,
                "timing_seconds": result.timing_seconds,
                "warnings": list(result.warnings),
            })
        else:
            info.update({
                "reward_raw": reward_raw if reward_raw is not None else 0.0,
                "reward_normalized": reward_normalized if reward_normalized is not None else 0.0,
            })

        return info

    # ---- Utility / inspection ----

    def get_state(self) -> _EnvState:
        """Return the internal episode state (for testing and debugging)."""
        return self._state

    def get_current_profit(self) -> float:
        """Return the current network profit η_t."""
        return self._state.profit_history[-1] if self._state.profit_history else 0.0

    def is_terminal(self) -> bool:
        """Check if the current episode is in a terminal state."""
        return self._terminated or self._truncated

    def get_last_mcf_result(self) -> Optional[MCFResult]:
        """Return the most recent MCF evaluation result."""
        return self._state.last_mcf_result


# ---------------------------------------------------------------------------
# Convenience: make ServiceAction the module-level name expected by tests
# ---------------------------------------------------------------------------

__all__ = ["LSNDPEnv", "ServiceAction", "ServiceValidationError", "_EnvState"]
