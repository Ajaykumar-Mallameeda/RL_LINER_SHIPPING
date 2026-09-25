"""
PPO Rollout Buffer.

Stores trajectories collected during PPO training. Preserves the distinction
between POLICY ACTION (what the policy sampled) and EXECUTED ENVIRONMENT
ACTION (what the environment actually executed, after TSP reordering, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class TrajectoryStep:
    """
    Single step in a PPO trajectory.

    Stores everything needed for a PPO update. The key invariant is that
    old_log_prob corresponds to the action that was actually SAMPLED by the
    policy, not the executed action (which may differ due to TSP reordering,
    fallback repairs, etc.).

    Attributes
    ----------
    state_repr : Any
        State representation consumed by the neural network (GraphTensors).
    policy_id : str
        Identifier for which policy produced this step ("encoder_only" or "encoder_decoder").
    action : Any
        The policy's raw action representation (for debugging/inspection).
    executed_action : Any
        The action actually executed by the environment (after TSP, validation, etc.).
    reward : float
        Environment reward R_{t+1}.
    done : bool
        Whether the episode terminated naturally.
    truncated : bool
        Whether the episode was truncated (safety cap reached).
    old_log_prob : torch.Tensor, scalar
        log P(action | state) under the policy at sampling time.
    old_value : torch.Tensor, scalar
        Value estimate V(s_t) at sampling time.
    entropy : torch.Tensor, scalar
        Policy entropy at sampling time (for diagnostics/bonus).
    fallback_applied : bool
        For P8: True if a deterministic fallback was applied to repair the draw.
    decoded_port_sequence : list[str]
        For P9: Ports in decoder selection order (for debugging).
    executed_port_sequence : list[str]
        For P9: Ports after P6 TSP reordering.
    """

    state_repr: Any
    policy_id: str
    action: Any
    executed_action: Any
    reward: float
    done: bool
    truncated: bool
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    entropy: torch.Tensor
    fallback_applied: bool = False
    decoded_port_sequence: List[str] = field(default_factory=list)
    executed_port_sequence: List[str] = field(default_factory=list)


class PPOBuffer:
    """
    Research-grade trajectory/rollout buffer for PPO training.

    The buffer stores complete trajectories and provides methods for:
      - Inserting steps during rollout collection
      - Computing returns and advantages
      - Sampling minibatches for PPO updates
      - Checking memory usage and capacity

    IMPORTANT: The buffer preserves the distinction between:
      - POLICY ACTION: What the policy sampled (used for log_prob computation)
      - EXECUTED ACTION: What the environment received (used for reward)
    """

    def __init__(self) -> None:
        self._steps: List[TrajectoryStep] = []
        self._episode_trajectories: List[List[TrajectoryStep]] = []

    def add_step(
        self,
        state_repr: Any,
        policy_id: str,
        action: Any,
        executed_action: Any,
        reward: float,
        done: bool,
        truncated: bool,
        old_log_prob: torch.Tensor,
        old_value: torch.Tensor,
        entropy: torch.Tensor,
        fallback_applied: bool = False,
        decoded_port_sequence: Optional[List[str]] = None,
        executed_port_sequence: Optional[List[str]] = None,
    ) -> None:
        """
        Add a single step to the buffer.

        Parameters
        ----------
        state_repr : Any
            GraphTensors or equivalent state representation.
        policy_id : str
            "encoder_only" or "encoder_decoder".
        action : Any
            Raw policy output (EncoderOnlyOutput or EncoderDecoderOutput).
        executed_action : Any
            ServiceAction actually passed to environment.
        reward : float
            Environment reward.
        done : bool
            Episode termination flag.
        truncated : bool
            Episode truncation flag.
        old_log_prob : torch.Tensor
            Scalar log-probability of the sampled action.
        old_value : torch.Tensor
            Scalar value estimate.
        entropy : torch.Tensor
            Scalar entropy value.
        fallback_applied : bool
            True if P8 fallback was applied (sample should be excluded).
        decoded_port_sequence : list[str], optional
            For P9: decoder's port selection order.
        executed_port_sequence : list[str], optional
            For P9: P6 TSP-ordered port sequence.
        """
        step = TrajectoryStep(
            state_repr=state_repr,
            policy_id=policy_id,
            action=action,
            executed_action=executed_action,
            reward=reward,
            done=done,
            truncated=truncated,
            old_log_prob=old_log_prob,
            old_value=old_value,
            entropy=entropy,
            fallback_applied=fallback_applied,
            decoded_port_sequence=list(decoded_port_sequence or []),
            executed_port_sequence=list(executed_port_sequence or []),
        )
        self._steps.append(step)

    def end_episode(self) -> List[TrajectoryStep]:
        """
        Mark current episode as complete. Returns the episode trajectory.

        In a multi-environment setup, this is called after each env completes.
        For single-env synchronous training, call after every done/truncated.
        """
        if not self._steps:
            return []
        episode = list(self._steps)
        self._episode_trajectories.append(episode)
        self._steps.clear()
        return episode

    def clear(self) -> None:
        """Clear all stored data."""
        self._steps.clear()
        self._episode_trajectories.clear()

    def __len__(self) -> int:
        return len(self._steps) + sum(len(ep) for ep in self._episode_trajectories)

    def get_all_steps(self) -> List[TrajectoryStep]:
        """Return all steps including in-progress episode."""
        result = list(self._steps)
        for ep in self._episode_trajectories:
            result.extend(ep)
        return result

    def get_completed_episodes(self) -> List[List[TrajectoryStep]]:
        """Return list of completed episode trajectories."""
        return list(self._episode_trajectories)

    def has_fallback_samples(self) -> bool:
        """Check if any stored steps have fallback_applied=True."""
        for step in self.get_all_steps():
            if step.fallback_applied:
                return True
        return False

    def filter_non_fallback(self) -> List[TrajectoryStep]:
        """
        Return steps excluding those with fallback_applied=True.

        Per P10.5 contract: fallback-repaired samples must not be treated as
        ordinary on-policy PPO samples unless P10 explicitly accounts for the
        repair transformation. The safe default is to exclude them.
        """
        return [s for s in self.get_all_steps() if not s.fallback_applied]
