"""
P3 – Expanded (proxy-node) graph for MCF evaluation.

Implements the paper's expanded graph construction (Algorithm 1, Appendix B,
Figure 4). The expanded graph G' = (N', E') adds service-specific proxy nodes
and handling-cost edges to the physical port network.

Node types:
  - Physical port node:   str (UNLOCODE), e.g. "DEBRV"
  - Proxy loading node:   ProxyNode("DEBRV", service_id=0)
  - Proxy offloading node: ProxyNode("DEBRV", service_id=0) — same physical, different role

Edge types and weights:
  +------------------+---------+----------+---------------------------+
  | Edge type        | From    | To       | Weight                    |
  +------------------+---------+----------+---------------------------+
  | Loading          | p       | p_s      | p_l (cost_per_full)       |
  | Offloading       | p_s     | q        | p_l (cost_per_full)       |
  | Transshipment    | p_{s'}  | p_{s''}  | p_t (cost_per_full_transfer)|
  | Service transit  | p_s     | q_s      | 0                         |
  +------------------+---------+----------+---------------------------+
  | Capacity         | loading/offload/transship | inf |
  | Capacity         | service transit            | n_vs * v_cap |
  +------------------+-----------------------------+-----+
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# Proxy node – unique identity per (physical_port, service_id)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, order=True)
class ProxyNode:
    """
    A service-specific proxy node representing "port p on service s".

    Implements __hash__ and __eq__ via the frozen dataclass default.
    Order is by (port_code, service_id) for deterministic tie-breaking.
    """
    port_code: str
    service_id: int

    def __str__(self) -> str:
        return f"{self.port_code}[s{self.service_id}]"

    def __repr__(self) -> str:
        return f"ProxyNode({self.port_code!r}, {self.service_id})"


# ---------------------------------------------------------------------------
# Edge metadata
# ---------------------------------------------------------------------------

@dataclass
class GraphEdge:
    """An edge in the expanded graph with its attributes."""
    u: object  # ProxyNode or str (physical port)
    v: object  # ProxyNode or str (physical port)
    weight: float
    capacity: float  # infinity represented as float('inf')
    edge_type: str  # "loading" | "offloading" | "transshipment" | "service"


# ---------------------------------------------------------------------------
# ExpandedGraph
# ---------------------------------------------------------------------------

class ExpandedGraph:
    """
    Build and manage the paper's expanded/proxy-node graph.

    Parameters
    ----------
    instance :
        Loaded LINERLIBInstance providing port cost parameters.
    services :
        List of ServiceDefinition objects (see below).
    distances_by_pair :
        Dict[(origin, dest)] -> DistanceArc for cost lookup.
    vessel_types :
        Dict[vessel_class] -> VesselType for capacity lookup.
    vessel_requirements :
        Dict[service_id] -> Dict[vessel_class, n_vs] computed by NDP.

    Usage
    -----
    >>> eg = ExpandedGraph(instance, services, dist_map, vtypes, vreqs)
    >>> eg.build()
    >>> result = eg.shortest_path(origin_proxy, dest_physical, capacities)
    """

    def __init__(
        self,
        instance,
        services,
        distances_by_pair: Dict[tuple, any],
        vessel_types: Dict[str, any],
        vessel_requirements: Dict[str, Dict[str, float]],
    ) -> None:
        self._instance = instance
        self._services = services
        self._dist = distances_by_pair
        self._vessel_types = vessel_types
        self._vreqs = vessel_requirements

        self._nodes: Set[object] = set()
        self._edges: List[GraphEdge] = []
        self._adj: Dict[object, List[Tuple[object, int]]] = defaultdict(list)
        self._edge_idx: Dict[Tuple[object, object], int] = {}
        self._node_to_ports: Dict[object, str] = {}
        self._port_proxies: Dict[str, Set[int]] = defaultdict(set)

    # ---- builders ----

    def build(self) -> "ExpandedGraph":
        """
        Construct the full expanded graph from services and instance data.

        Steps:
        1. Create physical port nodes.
        2. For each service visiting a port, create a proxy node.
        3. Add loading edges (physical → proxy) weighted p_l, cap = ∞.
        4. Add offloading edges (proxy → physical) weighted p_l, cap = ∞.
        5. Add transshipment edges (proxy_s' → proxy_s'') weighted p_t, cap = ∞.
        6. Add service transit edges (proxy_p → proxy_q) weighted 0, cap = n_vs * v_cap.
        """
        self._nodes.clear()
        self._edges.clear()
        self._adj.clear()
        self._edge_idx.clear()

        ports = self._instance.ports

        # Step 1: physical port nodes
        for pcode in ports:
            self._nodes.add(pcode)
            self._node_to_ports[pcode] = pcode

        # Step 2-6: iterate over services
        for sid, svc in enumerate(self._services):
            seq = svc.port_sequence
            vclass = svc.vessel_class
            vt = self._vessel_types.get(vclass)
            if vt is None:
                continue

            n_vs = self._vreqs.get(svc.service_id, {}).get(vclass, 0.0)
            edge_cap = n_vs * vt.capacity_ffe  # FFE/week

            # Create proxy nodes and add loading/offloading/service-transit edges
            prev_proxy = None
            for i, p_from in enumerate(seq):
                p_to = seq[(i + 1) % len(seq)]  # cyclic

                p_from_proxy = ProxyNode(p_from, sid)
                p_to_proxy = ProxyNode(p_to, sid)
                self._nodes.add(p_from_proxy)
                self._nodes.add(p_to_proxy)
                self._node_to_ports[p_from_proxy] = p_from
                self._node_to_ports[p_to_proxy] = p_to
                self._port_proxies[p_from].add(sid)
                self._port_proxies[p_to].add(sid)

                # Loading edge: physical p_from → proxy p_from[sid]
                pl = ports[p_from].cost_per_full if p_from in ports else 0.0
                self._add_edge(GraphEdge(
                    u=p_from, v=p_from_proxy,
                    weight=pl, capacity=float("inf"),
                    edge_type="loading",
                ))

                # Service transit edge: proxy p_from[sid] → proxy p_to[sid]
                self._add_edge(GraphEdge(
                    u=p_from_proxy, v=p_to_proxy,
                    weight=0.0, capacity=edge_cap,
                    edge_type="service",
                ))

                # Offloading edge: proxy p_to[sid] → physical p_to
                pt_load = ports[p_to].cost_per_full if p_to in ports else 0.0
                self._add_edge(GraphEdge(
                    u=p_to_proxy, v=p_to,
                    weight=pt_load, capacity=float("inf"),
                    edge_type="offloading",
                ))

                # Transshipment edges: from every other service's proxy at p_to
                pt_cost = ports[p_to].cost_per_full_transfer if p_to in ports else 0.0
                for other_sid, other_svc in enumerate(self._services):
                    if other_sid == sid:
                        continue
                    # Check if other service also visits p_to
                    if p_to in other_svc.visited_ports:
                        other_proxy = ProxyNode(p_to, other_sid)
                        self._add_edge(GraphEdge(
                            u=ProxyNode(p_to, sid), v=other_proxy,
                            weight=pt_cost, capacity=float("inf"),
                            edge_type="transshipment",
                        ))

            # Also handle transshipment FROM p_from to other services at p_from
            # (cargo arriving on this service can transfer to another at origin)
            # Note: these are already handled when the other service's loop processes
            # its own proxies. We only need outbound transshipment from each proxy.

        return self

    def _add_edge(self, edge: GraphEdge) -> None:
        """Add an edge to the adjacency structure."""
        key = (edge.u, edge.v)
        idx = len(self._edges)
        self._edges.append(edge)
        self._adj[edge.u].append((edge.v, idx))
        self._edge_idx[key] = idx

    # ---- accessors ----

    @property
    def nodes(self) -> Set[object]:
        return set(self._nodes)

    @property
    def edges(self) -> List[GraphEdge]:
        return list(self._edges)

    @property
    def adj(self) -> Dict[object, List[Tuple[object, int]]]:
        return dict(self._adj)

    @property
    def num_nodes(self) -> int:
        return len(self._nodes)

    @property
    def num_edges(self) -> int:
        return len(self._edges)

    def get_edge_type(self, u: object, v: object) -> Optional[str]:
        """Return the edge type string, or None if no direct edge exists."""
        idx = self._edge_idx.get((u, v))
        if idx is not None:
            return self._edges[idx].edge_type
        return None

    def get_capacity(self, u: object, v: object) -> float:
        """Return residual capacity of edge (u, v)."""
        idx = self._edge_idx.get((u, v))
        if idx is not None:
            return self._edges[idx].capacity
        return 0.0

    def get_weight(self, u: object, v: object) -> float:
        """Return edge weight."""
        idx = self._edge_idx.get((u, v))
        if idx is not None:
            return self._edges[idx].weight
        return float("inf")

    def get_service_at_port(self, proxy: ProxyNode) -> int:
        """Return the service ID for a proxy node."""
        return proxy.service_id

    def get_physical_port(self, node: object) -> str:
        """Return the physical port UNLOCODE for any node."""
        return self._node_to_ports.get(node, str(node))

    def get_loading_unloading_cost(self, pcode: str) -> float:
        """Get p_l for a physical port."""
        p = self._instance.ports.get(pcode)
        return p.cost_per_full if p and p.cost_per_full is not None else 0.0

    def get_transshipment_cost(self, pcode: str) -> float:
        """Get p_t for a physical port."""
        p = self._instance.ports.get(pcode)
        return p.cost_per_full_transfer if p and p.cost_per_full_transfer is not None else 0.0


# ---------------------------------------------------------------------------
# Service definition helper (minimal)
# ---------------------------------------------------------------------------

@dataclass
class ServiceDefinition:
    """
    Minimal service representation for MCF evaluation.

    Parameters
    ----------
    service_id :
        Unique integer ID assigned by the caller.
    vessel_class :
        Primary vessel class name.
    port_sequence :
        Ordered list of port UNLOCODEs forming a cyclic rotation.
    """
    service_id: int
    vessel_class: str
    port_sequence: List[str]

    @property
    def visited_ports(self) -> Set[str]:
        return set(self.port_sequence)

    @property
    def service_id_str(self) -> str:
        return f"svc_{self.service_id}"
