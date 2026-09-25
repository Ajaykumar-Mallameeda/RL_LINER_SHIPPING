"""
P10 — Value function (critic) for PPO training.

The critic consumes an appropriate state representation and produces
V_theta(S_t). It is a P10-owned module — NOT inserted into P7/P8/P9.

Design:
  - Input: flattened state features from GraphTensors
  - Output: scalar value estimate V(s)
  - Architecture: MLP with configurable hidden layers
  - Consumes: port_features + vessel_features flattened (same info as P5)

This critic is shared between P8 and P9 policies — it sees the same
state representation regardless of which policy generated the action.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ValueFunction(nn.Module):
    """
    Scalar value function V_theta(s) for PPO training.

    Parameters
    ----------
    input_dim : int
        Dimension of the flattened state representation.
    hidden_dims : tuple[int, ...]
        Hidden layer dimensions. Default: (256, 128).
    activation : str
        Activation function: "relu" or "elu". Default: "relu".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        activation: str = "relu",
    ) -> None:
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "elu":
                layers.append(nn.ELU())
            else:
                raise ValueError(f"Unknown activation: {activation}")
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (batch, input_dim) or (input_dim,)
            Flattened state representation.

        Returns
        -------
        Tensor, shape (batch, 1) or (1,)
            Value estimate V(s).
        """
        return self.network(x)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for forward for clarity."""
        return self.forward(x)


def build_critic_from_graph(
    graph_dim: int,
    config: Optional[Any] = None,
) -> ValueFunction:
    """
    Build a critic that consumes the same features as the policy backbone.

    Parameters
    ----------
    graph_dim : int
        Dimension of the flattened graph state (port_features + vessel_features).
    config : dict, optional
        Configuration overrides for the critic.

    Returns
    -------
    ValueFunction
    """
    hidden_dims = (256, 128)
    if config and "hidden_dims" in config:
        hidden_dims = tuple(config["hidden_dims"])
    return ValueFunction(input_dim=graph_dim, hidden_dims=hidden_dims)
