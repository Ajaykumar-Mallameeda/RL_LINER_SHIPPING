"""
P7 — P5 → PyTorch tensor conversion boundary.

P5 (`state.representation.NeuralState`) owns numpy arrays and deterministic
indexing. P7 owns the conversion to torch tensors with explicit dtype/device
handling. This module is the ONLY place where that conversion happens, so that
device/dtype policy is stated once and cannot drift between call sites.

Design constraints (from the batch specification):

  * P7 must NOT recreate P5 preprocessing. Every feature value here is copied
    verbatim from `NeuralState`; nothing is recomputed or re-normalised.
  * Edge alignment must be provably preserved: `edge_index[:, i]`,
    `static_edge_features[:, i]` and `dynamic_edge_features[:, i]` must all
    refer to the same edge. This module asserts that invariant on every
    conversion (see `verify_edge_alignment`).
  * No hard-coded CUDA. Device defaults to CPU.
  * Node/edge ordering is never re-sorted here. P5 already fixes the order
    (ports alphabetically, edges by sorted (origin, dest)). Any reordering
    would silently break alignment with P5's index maps.

Evidence tags: [IMPLEMENTATION] unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from state.representation import NeuralState

# Torch is imported lazily-safe: a clear error is raised if it is missing,
# rather than an import failure at package import time.
try:  # pragma: no cover - exercised implicitly by the whole test suite
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "P7 requires PyTorch. Install with: pip install torch"
    ) from exc


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GLOBAL_NODE_OFFSET: int = 1  # [PAPER] Section 2.1 — one extra global node.

# Reserved edge-feature width: static(4) + dynamic(2 + |S|_max).
# [PAPER] Eq. 39 gives D_e = 6 + |S|, with |S| the MAXIMUM number of services;
# P5 fixes |S|_max = 100 (matching P4's safety cap). A live P5 state emits
# 4 + 2 + num_services_active columns, i.e. <= this width, and the GAT zero-pads
# the absent service rows.
RESERVED_EDGE_FEATURE_DIM: int = 4 + 2 + 100

_DTYPE_MAP: Dict[str, "torch.dtype"] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


# ---------------------------------------------------------------------------
# GraphTensors
# ---------------------------------------------------------------------------

@dataclass
class GraphTensors:
    """
    Torch view of a P5 `NeuralState`, ready for the P7 backbone.

    Every tensor is aligned on the edge dimension E:
        edge_index[:, i]            → (origin_node, dest_node) of edge i
        static_edge_features[:, i]  → static features of edge i
        dynamic_edge_features[:, i] → dynamic features of edge i

    Attributes
    ----------
    node_features : Tensor, shape (P+1, 2)
        [PAPER] Eq. 37-38. Row P is the global node, which is [0, 0].
    static_edge_features : Tensor, shape (4, E)
        [PAPER] Eq. 39 static block.
    dynamic_edge_features : Tensor, shape (2 + num_services, E)
        [PAPER] Eq. 39 dynamic block. Row count grows with service count.
    edge_index : LongTensor, shape (2, E)
        Derived from the origin/destination index rows of
        `static_edge_features` — never re-derived from any other source, so
        alignment with P5 is structural rather than assumed.
    vessel_features : Tensor, shape (V, 11)
        [PAPER] Appendix A.1.
    port_codes : list[str]
        Port UNLOCODEs in node order (index i → port_codes[i]).
    vessel_classes : list[str]
        Vessel class names in vessel index order.
    num_ports : int
        P — number of physical ports (excludes the global node).
    num_nodes : int
        P + 1 — includes the global node.
    num_edges : int
        E.
    num_vessel_classes : int
        V.
    num_services : int
        |S| currently active (width of the dynamic service-membership block).
    instance_name : str
    device : torch.device
    dtype : torch.dtype
    """

    node_features: "torch.Tensor"
    static_edge_features: "torch.Tensor"
    dynamic_edge_features: "torch.Tensor"
    edge_index: "torch.Tensor"
    vessel_features: "torch.Tensor"
    port_codes: List[str]
    vessel_classes: List[str]
    num_ports: int
    num_nodes: int
    num_edges: int
    num_vessel_classes: int
    num_services: int
    instance_name: str
    device: "torch.device"
    dtype: "torch.dtype"

    # ---- convenience ----

    def to(self, device: Any) -> "GraphTensors":
        """Return a copy of this bundle moved to `device`."""
        dev = torch.device(device)
        return GraphTensors(
            node_features=self.node_features.to(dev),
            static_edge_features=self.static_edge_features.to(dev),
            dynamic_edge_features=self.dynamic_edge_features.to(dev),
            edge_index=self.edge_index.to(dev),
            vessel_features=self.vessel_features.to(dev),
            port_codes=list(self.port_codes),
            vessel_classes=list(self.vessel_classes),
            num_ports=self.num_ports,
            num_nodes=self.num_nodes,
            num_edges=self.num_edges,
            num_vessel_classes=self.num_vessel_classes,
            num_services=self.num_services,
            instance_name=self.instance_name,
            device=dev,
            dtype=self.dtype,
        )

    def port_index(self, port_code: str) -> int:
        """Node index of a port UNLOCODE. Raises KeyError if absent."""
        try:
            return self.port_codes.index(port_code)
        except ValueError as exc:
            raise KeyError(
                f"Port {port_code!r} not in this graph "
                f"({self.num_ports} ports)."
            ) from exc

    def vessel_index(self, vessel_class: str) -> int:
        """Vessel index of a class name. Raises KeyError if absent."""
        try:
            return self.vessel_classes.index(vessel_class)
        except ValueError as exc:
            raise KeyError(
                f"Vessel class {vessel_class!r} not in this graph "
                f"({self.num_vessel_classes} classes)."
            ) from exc


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def _as_tensor(
    arr: np.ndarray,
    dtype: "torch.dtype",
    device: "torch.device",
) -> "torch.Tensor":
    """Convert a numpy array to a tensor, validating input shape/type."""
    if not isinstance(arr, np.ndarray):
        raise TypeError(
            f"Expected a numpy ndarray from P5, got {type(arr).__name__}."
        )
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2-D array, got shape {arr.shape}.")
    if np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        raise ValueError("Feature array contains non-finite values (NaN/Inf).")
    return torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype, device=device)


def neural_state_to_tensors(
    state: NeuralState,
    config: Optional[Any] = None,
    device: Optional[Any] = None,
    dtype: Optional[Any] = None,
) -> GraphTensors:
    """
    Convert a P5 `NeuralState` into aligned torch tensors.

    Parameters
    ----------
    state : NeuralState
        Output of `StateEncoder.encode`. Read-only; never mutated.
    config : ArchitectureConfig, optional
        Supplies default device/dtype when the explicit arguments are None.
    device : str or torch.device, optional
        Target device. Defaults to config.device, else "cpu".
    dtype : str or torch.dtype, optional
        Target float dtype. Defaults to config.dtype, else "float32".

    Returns
    -------
    GraphTensors

    Raises
    ------
    ValueError
        If the P5 state is structurally inconsistent — mismatched edge counts
        between the static and dynamic edge blocks, or an edge whose recorded
        endpoint index is out of range for the node tensor.
    """
    # ---- resolve device / dtype ----
    if dtype is None:
        dtype = getattr(config, "dtype", "float32")
    if isinstance(dtype, str):
        if dtype not in _DTYPE_MAP:
            raise ValueError(
                f"Unsupported dtype {dtype!r}; expected one of "
                f"{sorted(_DTYPE_MAP)}."
            )
        torch_dtype = _DTYPE_MAP[dtype]
    else:
        torch_dtype = dtype

    if device is None:
        device = getattr(config, "device", "cpu")
    torch_device = torch.device(device)

    # ---- validate P5 shapes before converting ----
    # P5 names the node tensor `port_features` (shape (P+1, 2), global node last).
    node_feats = state.port_features
    static_e = state.static_edge_features
    dynamic_e = state.dynamic_edge_features
    vessel_f = state.vessel_features

    # Type-check everything first: a wrong type must be reported as a type
    # error, not as a confusing AttributeError from a later shape check.
    for label, arr in (
        ("port_features", node_feats),
        ("static_edge_features", static_e),
        ("dynamic_edge_features", dynamic_e),
        ("vessel_features", vessel_f),
    ):
        if not isinstance(arr, np.ndarray):
            raise TypeError(
                f"P5 {label} must be a numpy ndarray, got "
                f"{type(arr).__name__}."
            )

    if static_e.shape[1] != dynamic_e.shape[1]:
        raise ValueError(
            "P5 state is inconsistent: static_edge_features has "
            f"{static_e.shape[1]} edges but dynamic_edge_features has "
            f"{dynamic_e.shape[1]}."
        )
    if vessel_f.shape[1] != 11:
        raise ValueError(
            f"P5 vessel_features must have 11 columns (Appendix A.1), got "
            f"{vessel_f.shape[1]}."
        )
    if node_feats.shape[1] != 2:
        raise ValueError(
            f"P5 port_features must have 2 columns (Eq. 37-38), got "
            f"{node_feats.shape[1]}."
        )

    num_nodes = int(node_feats.shape[0])
    num_ports = num_nodes - GLOBAL_NODE_OFFSET
    num_edges = int(static_e.shape[1])

    # ---- convert ----
    node_t = _as_tensor(node_feats, torch_dtype, torch_device)
    static_t = _as_tensor(static_e, torch_dtype, torch_device)
    dynamic_t = _as_tensor(dynamic_e, torch_dtype, torch_device)
    vessel_t = _as_tensor(vessel_f, torch_dtype, torch_device)

    # ---- derive edge_index FROM THE STATIC FEATURE ROWS ----
    # Rows 0 and 1 of the static block are, by P5's contract (Eq. 39), the
    # origin and destination node indices. Deriving edge_index from them
    # (rather than from any separate list) makes alignment structural.
    origin_idx = static_t[0].to(torch.long)
    dest_idx = static_t[1].to(torch.long)
    edge_index = torch.stack([origin_idx, dest_idx], dim=0)

    # ---- ordering maps from P5's deterministic index maps ----
    port_to_node: Dict[str, int] = state.indices["port_to_node"]
    vessel_to_vessel: Dict[str, int] = state.indices["vessel_to_vessel"]

    port_codes = [None] * num_ports  # type: ignore[list-item]
    for code, idx in port_to_node.items():
        if not 0 <= idx < num_ports:
            raise ValueError(
                f"P5 port_to_node maps {code!r} to index {idx}, out of range "
                f"for {num_ports} ports."
            )
        port_codes[idx] = code
    if any(c is None for c in port_codes):
        missing = [i for i, c in enumerate(port_codes) if c is None]
        raise ValueError(f"P5 port_to_node does not cover node indices {missing}.")

    vessel_classes = [None] * len(vessel_to_vessel)  # type: ignore[list-item]
    for name, idx in vessel_to_vessel.items():
        if not 0 <= idx < len(vessel_to_vessel):
            raise ValueError(
                f"P5 vessel_to_vessel maps {name!r} to index {idx}, out of "
                f"range for {len(vessel_to_vessel)} classes."
            )
        vessel_classes[idx] = name
    if any(c is None for c in vessel_classes):
        missing = [i for i, c in enumerate(vessel_classes) if c is None]
        raise ValueError(
            f"P5 vessel_to_vessel does not cover vessel indices {missing}."
        )

    bundle = GraphTensors(
        node_features=node_t,
        static_edge_features=static_t,
        dynamic_edge_features=dynamic_t,
        edge_index=edge_index,
        vessel_features=vessel_t,
        port_codes=port_codes,  # type: ignore[arg-type]
        vessel_classes=vessel_classes,  # type: ignore[arg-type]
        num_ports=num_ports,
        num_nodes=num_nodes,
        num_edges=num_edges,
        num_vessel_classes=len(vessel_classes),
        num_services=int(state.num_services),
        instance_name=str(state.instance_name),
        device=torch_device,
        dtype=torch_dtype,
    )

    # ---- enforce the alignment invariant before handing this out ----
    verify_edge_alignment(bundle)

    return bundle


# ---------------------------------------------------------------------------
# Invariant checks
# ---------------------------------------------------------------------------

def verify_edge_alignment(bundle: GraphTensors) -> None:
    """
    Assert that edge_index, static and dynamic edge features are aligned.

    This is the P7.3 invariant. It is checked on construction and re-checked
    by the test suite. Raises ValueError with a precise diagnosis on failure.

    Checks performed
    ----------------
    1. All three edge-dimension tensors have the same width E.
    2. `edge_index[0]` equals the origin row of the static block.
    3. `edge_index[1]` equals the destination row of the static block.
    4. Every endpoint index is within [0, num_nodes).
    """
    E = bundle.num_edges
    if bundle.static_edge_features.shape[1] != E:
        raise ValueError(
            f"static_edge_features width {bundle.static_edge_features.shape[1]}"
            f" != num_edges {E}."
        )
    if bundle.dynamic_edge_features.shape[1] != E:
        raise ValueError(
            f"dynamic_edge_features width "
            f"{bundle.dynamic_edge_features.shape[1]} != num_edges {E}."
        )
    if bundle.edge_index.shape != (2, E):
        raise ValueError(
            f"edge_index shape {tuple(bundle.edge_index.shape)} != (2, {E})."
        )

    if E == 0:
        return

    origin = bundle.static_edge_features[0]
    dest = bundle.static_edge_features[1]
    if not torch.equal(bundle.edge_index[0], origin.to(torch.long)):
        raise ValueError(
            "edge_index origin row is not aligned with "
            "static_edge_features row 0."
        )
    if not torch.equal(bundle.edge_index[1], dest.to(torch.long)):
        raise ValueError(
            "edge_index destination row is not aligned with "
            "static_edge_features row 1."
        )

    if int(bundle.edge_index.min()) < 0 or int(bundle.edge_index.max()) >= bundle.num_nodes:
        raise ValueError(
            f"edge_index endpoints must lie in [0, {bundle.num_nodes}); got "
            f"range [{int(bundle.edge_index.min())}, "
            f"{int(bundle.edge_index.max())}]."
        )


def edge_to_ports(
    bundle: GraphTensors,
    edge_idx: int,
) -> Tuple[str, str]:
    """
    Resolve edge index i to its (origin_port, destination_port) UNLOCODEs.

    Provided so tests can verify alignment against P5's `od_to_edge` map by an
    independent route (names) rather than by re-reading the same index row.
    """
    if not 0 <= edge_idx < bundle.num_edges:
        raise IndexError(
            f"Edge index {edge_idx} out of range for {bundle.num_edges} edges."
        )
    o_idx = int(bundle.edge_index[0, edge_idx])
    d_idx = int(bundle.edge_index[1, edge_idx])
    return bundle.port_codes[o_idx], bundle.port_codes[d_idx]
