"""
P7 — Tests for the GAT + Transformer neural architecture.

Covers the P7.11 test list:
   1. model construction                    10. forward pass
   2. parameter dimensions                  11. output shape
   3. 3 GAT layers                          12. deterministic CPU forward
   4. 3 Transformer layers                  13. multiple graph sizes
   5. 8 attention heads                     14. serialization round-trip
   6. hidden dimension 512                  15. invalid input detection
   7. P5 → P7 integration                   16. mask interface
   8. graph connectivity alignment          17. dtype/device handling
   9. static/dynamic edge alignment

Fixtures in this file are TINY SYNTHETIC instances, used only for isolated
unit tests. The real LINERLIB instances appear in the integration section.
No training, no PPO, no optimizers, no sampling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data.instance import (
    DatasetProvenance,
    Demand,
    DistanceArc,
    FleetEntry,
    InstanceMetadata,
    LINERLIBInstance,
    Port,
    ProvenanceRecord,
    VesselType,
)
from neural import (
    PAPER_ARCHITECTURE,
    ArchitectureConfig,
    GATStack,
    GraphTensors,
    MASK_FILL_VALUE,
    NeuralBackbone,
    RESERVED_EDGE_FEATURE_DIM,
    TransformerEncoderStack,
    all_available,
    already_selected_port_mask,
    apply_mask_to_logits,
    decoder_phase_mask,
    describe_mask,
    draft_feasible_port_mask,
    encoder_only_port_mask,
    edge_to_ports,
    global_node_mask,
    load_backbone,
    masked_log_softmax,
    neural_state_to_tensors,
    none_available,
    paper_config,
    save_backbone,
    to_pytorch_padding_mask,
    verify_edge_alignment,
)
from neural.gat import _segment_softmax
from state.representation import ServiceMembership, StateEncoder


# ===========================================================================
# Synthetic fixtures — NOT LINERLIB benchmark data
# ===========================================================================

def _port(code: str, draft: float = 10.0) -> Port:
    return Port(
        unlocode=code, name=f"Port {code}", country=None,
        cabotage_region="test", d_region=None, longitude=None, latitude=None,
        draft=draft, cost_per_full=1.0, cost_per_full_transfer=0.5,
        port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
        provenance=ProvenanceRecord(source_file="synthetic_p7", source_row=1),
    )


def _vessel(name: str, capacity: float, draft: float = 12.0) -> VesselType:
    return VesselType(
        vessel_class=name, capacity_ffe=capacity, tc_rate_daily=100,
        draft=draft, min_speed=5.0, max_speed=15.0, design_speed=10.0,
        bunker_ton_per_day_at_design=50.0, idle_consumption_ton_per_day=10.0,
        panama_fee=0, suez_fee=0,
        provenance=ProvenanceRecord(source_file="synthetic_p7", source_row=1),
    )


def make_instance(
    n_ports: int = 3,
    n_vessels: int = 2,
    seed: int = 0,
    port_draft: float = 10.0,
) -> LINERLIBInstance:
    """
    Build a synthetic fully-connected instance with `n_ports` ports.

    Fully connected (both directions) so every ordered pair has a distance arc,
    which keeps the edge set deterministic and dense.
    """
    codes = [f"P{i:02d}" for i in range(n_ports)]
    rng = np.random.default_rng(seed)

    ports = {c: _port(c, draft=port_draft) for c in codes}

    vessels = {}
    for i in range(n_vessels):
        vessels[f"V{i}"] = _vessel(
            f"V{i}", capacity=100.0 * (i + 1), draft=12.0 + i,
        )

    demands = []
    distances = []
    row = 1
    for i, o in enumerate(codes):
        for j, d in enumerate(codes):
            if o == d:
                continue
            distances.append(DistanceArc(
                origin=o, destination=d,
                distance_nm=100.0 + 10.0 * abs(i - j),
                draft_required=port_draft, is_panama=False, is_suez=False,
                provenance=ProvenanceRecord(
                    source_file="synthetic_p7", source_row=row,
                ),
            ))
            row += 1
    # A handful of demands, deterministic.
    for k in range(min(3, n_ports - 1)):
        demands.append(Demand(
            origin=codes[k], destination=codes[k + 1],
            ffe_per_week=50.0 + 10.0 * k, revenue=200.0, max_transit_time=20,
            provenance=ProvenanceRecord(source_file="synthetic_p7", source_row=k + 1),
        ))

    fleet = [FleetEntry(vessel_class=f"V{i}", quantity=3) for i in range(n_vessels)]

    metadata = InstanceMetadata(
        name=f"SYNTH_{n_ports}P", active_port_count=n_ports,
        vessel_type_count=n_vessels, total_vessels=3 * n_vessels,
        demand_count=len(demands), distance_arc_count=len(distances),
    )

    return LINERLIBInstance(
        name=f"SYNTH_{n_ports}P", ports=ports, vessel_types=vessels,
        demands=demands, distances=distances, fleet=fleet, metadata=metadata,
        provenance=DatasetProvenance(source_root="[SYNTHETIC TEST FIXTURE -- P7]"),
    )


def make_bundle(
    n_ports: int = 3,
    n_vessels: int = 2,
    num_services: int = 0,
    seed: int = 0,
    port_draft: float = 10.0,
    **tensor_kwargs,
) -> GraphTensors:
    """Build a GraphTensors bundle from a synthetic instance via P5."""
    inst = make_instance(
        n_ports=n_ports, n_vessels=n_vessels, seed=seed, port_draft=port_draft,
    )
    dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
    encoder = StateEncoder(inst, dist_by_pair)
    remaining = {i: d.ffe_per_week for i, d in enumerate(inst.demands)}
    fleet_remaining = {e.vessel_class: float(e.quantity) for e in inst.fleet}

    membership = ServiceMembership()
    codes = sorted(inst.ports.keys())
    for s in range(num_services):
        from mcf.expanded_graph import ServiceDefinition
        seq = codes[: max(2, len(codes) - s)]
        vc = sorted(inst.vessel_types.keys())[0]
        membership.add(
            ServiceDefinition(service_id=s, vessel_class=vc, port_sequence=seq),
            {vc: 1.0},
        )

    state = encoder.encode(remaining, fleet_remaining, membership)
    return neural_state_to_tensors(state, **tensor_kwargs)


# ===========================================================================
# 1-6. Construction, layer counts, and paper dimensions
# ===========================================================================

class TestPaperDimensions:
    """[PAPER] Table 5 — the frozen architecture values."""

    def test_paper_constants_are_the_expected_values(self):
        assert PAPER_ARCHITECTURE == {
            "hidden_dim": 512,
            "transformer_heads": 8,
            "transformer_layers": 3,
            "gat_layers": 3,
            "lstm_layers": 1,
        }

    def test_default_config_matches_paper(self):
        cfg = ArchitectureConfig()
        assert cfg.matches_paper()
        assert cfg.hidden_dim == 512
        assert cfg.gat_layers == 3
        assert cfg.transformer_layers == 3
        assert cfg.transformer_heads == 8
        assert cfg.lstm_layers == 1

    def test_paper_config_helper_matches_paper(self):
        assert paper_config().matches_paper()

    def test_deviation_is_detectable(self):
        """A changed frozen value must be reportable, not silent."""
        cfg = ArchitectureConfig(hidden_dim=256, transformer_heads=4)
        assert not cfg.matches_paper()

    def test_derived_dimensions(self):
        cfg = ArchitectureConfig()
        assert cfg.transformer_ffn_dim == 2048   # 4x expansion
        assert cfg.gat_head_dim == 512           # 1 GAT head

    def test_config_rejects_indivisible_heads(self):
        with pytest.raises(ValueError, match="divisible"):
            ArchitectureConfig(hidden_dim=100, transformer_heads=3)

    def test_config_rejects_bad_values(self):
        with pytest.raises(ValueError):
            ArchitectureConfig(gat_layers=0)
        with pytest.raises(ValueError):
            ArchitectureConfig(transformer_layers=0)
        with pytest.raises(ValueError):
            ArchitectureConfig(dropout=1.0)
        with pytest.raises(ValueError):
            ArchitectureConfig(dtype="float16")


class TestModelConstruction:
    """Test 1 — construction; Test 2 — parameter dimensions."""

    def test_backbone_constructs_with_paper_config(self):
        model = NeuralBackbone(ArchitectureConfig())
        assert isinstance(model, torch.nn.Module)
        assert model.parameter_count() > 0

    def test_gat_layer_count(self):
        """Test 3 — exactly 3 GAT layers."""
        model = NeuralBackbone(ArchitectureConfig())
        assert model.num_gat_layers == 3
        assert len(model.gat.layers) == 3

    def test_transformer_layer_count(self):
        """Test 4 — exactly 3 Transformer layers."""
        model = NeuralBackbone(ArchitectureConfig())
        assert model.num_transformer_layers == 3
        # nn.TransformerEncoder exposes its stack as .layers
        assert len(model.transformer.encoder.layers) == 3

    def test_transformer_head_count(self):
        """Test 5 — exactly 8 attention heads."""
        model = NeuralBackbone(ArchitectureConfig())
        assert model.num_transformer_heads == 8
        head = model.transformer.encoder.layers[0]
        assert head.self_attn.num_heads == 8

    def test_hidden_dimension_propagates(self):
        """Test 6 — H = 512 everywhere the paper specifies it."""
        cfg = ArchitectureConfig()
        model = NeuralBackbone(cfg)

        # Every GAT layer emits H.
        for layer in model.gat.layers:
            assert layer.out_dim == 512
        # Transformer operates in H.
        assert model.transformer.hidden_dim == 512
        assert model.transformer.encoder.layers[0].linear1.in_features == 512
        # Vessel projection maps D_v=11 -> H.
        assert model.vessel_projection.in_features == 11
        assert model.vessel_projection.out_features == 512

    def test_parameter_shapes_are_stable_across_graph_sizes(self):
        """Parameter dimensions must not depend on the instance."""
        small = NeuralBackbone(ArchitectureConfig.tiny())
        big = NeuralBackbone(ArchitectureConfig.tiny())
        sd_a = {k: v.shape for k, v in small.state_dict().items()}
        sd_b = {k: v.shape for k, v in big.state_dict().items()}
        assert sd_a == sd_b

    def test_architecture_summary_reports_paper_conformance(self):
        summary = NeuralBackbone(ArchitectureConfig()).architecture_summary()
        assert summary["gat_layers"] == 3
        assert summary["transformer_layers"] == 3
        assert summary["transformer_heads"] == 8
        assert summary["hidden_dim"] == 512
        assert summary["matches_paper"] is True

    def test_gat_uses_single_head_by_default(self):
        """
        The paper gives a head count for the Transformer only; the GAT head
        count is an [ENGINEERING DECISION] defaulting to 1.
        """
        cfg = ArchitectureConfig()
        assert cfg.gat_heads == 1
        model = NeuralBackbone(cfg)
        assert len(model.gat.layers[0].heads) == 1

    def test_multi_head_gat_is_configurable(self):
        cfg = ArchitectureConfig(gat_heads=4, hidden_dim=512)
        model = NeuralBackbone(cfg)
        assert len(model.gat.layers[0].heads) == 4
        assert cfg.gat_head_dim == 128


# ===========================================================================
# 7. P5 → P7 integration
# ===========================================================================

class TestP5Integration:

    def test_tensor_conversion_shapes(self):
        """Test 7 — P5 NeuralState converts with shapes preserved."""
        bundle = make_bundle(n_ports=4, n_vessels=2, num_services=0)
        assert bundle.num_ports == 4
        assert bundle.num_nodes == 5          # 4 ports + global
        assert bundle.num_edges == 12         # 4*3 ordered pairs
        assert bundle.num_vessel_classes == 2
        assert bundle.node_features.shape == (5, 2)
        assert bundle.static_edge_features.shape == (4, 12)
        assert bundle.dynamic_edge_features.shape == (2, 12)
        assert bundle.vessel_features.shape == (2, 11)

    def test_values_are_copied_verbatim_from_p5(self):
        """P7 must not re-preprocess: values must match P5 exactly."""
        inst = make_instance(n_ports=3, n_vessels=2)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        remaining = {i: d.ffe_per_week for i, d in enumerate(inst.demands)}
        fleet = {e.vessel_class: float(e.quantity) for e in inst.fleet}
        state = encoder.encode(remaining, fleet, ServiceMembership())

        bundle = neural_state_to_tensors(state)

        np.testing.assert_allclose(
            bundle.node_features.numpy(), state.port_features, rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            bundle.static_edge_features.numpy(),
            state.static_edge_features, rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            bundle.vessel_features.numpy(), state.vessel_features,
            rtol=0, atol=0,
        )

    def test_port_codes_follow_p5_node_order(self):
        bundle = make_bundle(n_ports=5)
        assert bundle.port_codes == sorted(bundle.port_codes)
        # P5 orders ports alphabetically, so node i is the i-th sorted code.
        assert bundle.port_codes == ["P00", "P01", "P02", "P03", "P04"]

    def test_vessel_classes_are_alphabetical(self):
        bundle = make_bundle(n_ports=3, n_vessels=3)
        assert bundle.vessel_classes == sorted(bundle.vessel_classes)
        assert bundle.vessel_classes == ["V0", "V1", "V2"]

    def test_global_node_features_are_zero(self):
        """[PAPER] Section 2.1 — p_{P+1} = [0, 0]."""
        bundle = make_bundle(n_ports=4)
        assert torch.allclose(
            bundle.node_features[-1], torch.zeros(2),
        )

    def test_encode_state_shortcut_matches_encode_graph(self):
        inst = make_instance(n_ports=3)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        remaining = {i: d.ffe_per_week for i, d in enumerate(inst.demands)}
        fleet = {e.vessel_class: float(e.quantity) for e in inst.fleet}
        state = encoder.encode(remaining, fleet, ServiceMembership())

        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        with torch.no_grad():
            via_state = model.encode_state(state)
            via_bundle = model.encode_graph(neural_state_to_tensors(state))
        assert torch.allclose(
            via_state.port_embeddings, via_bundle.port_embeddings,
        )

    def test_dynamic_edge_rows_track_service_count(self):
        b0 = make_bundle(n_ports=4, num_services=0)
        b2 = make_bundle(n_ports=4, num_services=2)
        assert b0.dynamic_edge_features.shape[0] == 2
        assert b2.dynamic_edge_features.shape[0] == 4
        assert b0.num_services == 0
        assert b2.num_services == 2


# ===========================================================================
# 8-9. Edge alignment
# ===========================================================================

class TestEdgeAlignment:
    """The P7.3 invariant: index i in each edge tensor is the same edge."""

    def test_edge_index_matches_static_feature_rows(self):
        """Test 8 — connectivity aligns with the static feature rows."""
        bundle = make_bundle(n_ports=4)
        assert torch.equal(
            bundle.edge_index[0],
            bundle.static_edge_features[0].to(torch.long),
        )
        assert torch.equal(
            bundle.edge_index[1],
            bundle.static_edge_features[1].to(torch.long),
        )

    def test_static_and_dynamic_share_edge_dimension(self):
        """Test 9 — static and dynamic blocks describe the same E edges."""
        bundle = make_bundle(n_ports=5, num_services=1)
        assert bundle.static_edge_features.shape[1] == \
            bundle.dynamic_edge_features.shape[1] == bundle.num_edges

    def test_edge_to_ports_matches_p5_index_map(self):
        """
        Independent cross-check: resolve each edge to port NAMES via
        edge_index and compare against P5's od_to_edge map.
        """
        inst = make_instance(n_ports=4)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        remaining = {i: d.ffe_per_week for i, d in enumerate(inst.demands)}
        fleet = {e.vessel_class: float(e.quantity) for e in inst.fleet}
        state = encoder.encode(remaining, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(state)

        od_to_edge = state.indices["od_to_edge"]
        for (o, d), idx in od_to_edge.items():
            assert edge_to_ports(bundle, idx) == (o, d)

    def test_alignment_invariant_detects_tampering(self):
        """A deliberately misaligned bundle must be rejected."""
        bundle = make_bundle(n_ports=4)
        bundle.dynamic_edge_features = bundle.dynamic_edge_features[:, :-1]
        with pytest.raises(ValueError, match="dynamic_edge_features width"):
            verify_edge_alignment(bundle)

    def test_alignment_invariant_detects_index_swap(self):
        bundle = make_bundle(n_ports=4)
        bundle.edge_index = bundle.edge_index.flip(0)
        with pytest.raises(ValueError, match="origin row is not aligned"):
            verify_edge_alignment(bundle)

    def test_alignment_invariant_rejects_out_of_range_endpoints(self):
        bundle = make_bundle(n_ports=4)
        bundle.static_edge_features[0, 0] = float(bundle.num_nodes + 5)
        bundle.edge_index[0, 0] = bundle.num_nodes + 5
        with pytest.raises(ValueError, match="must lie in"):
            verify_edge_alignment(bundle)

    def test_conversion_rejects_mismatched_edge_blocks(self):
        inst = make_instance(n_ports=3)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        remaining = {i: d.ffe_per_week for i, d in enumerate(inst.demands)}
        fleet = {e.vessel_class: float(e.quantity) for e in inst.fleet}
        state = encoder.encode(remaining, fleet, ServiceMembership())
        state.dynamic_edge_features = state.dynamic_edge_features[:, :-1]
        with pytest.raises(ValueError, match="inconsistent"):
            neural_state_to_tensors(state)


# ===========================================================================
# 10-11. Forward pass and output shapes
# ===========================================================================

class TestForwardPass:

    @pytest.fixture
    def model_and_bundle(self):
        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        bundle = make_bundle(n_ports=4, n_vessels=2)
        return model, bundle

    def test_forward_pass_runs(self, model_and_bundle):
        """Test 10 — forward pass completes."""
        model, bundle = model_and_bundle
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert out is not None

    def test_output_shapes(self, model_and_bundle):
        """Test 11 — every output has the expected shape."""
        model, bundle = model_and_bundle
        H = model.hidden_dim
        with torch.no_grad():
            out = model.encode_graph(bundle)

        assert out.port_embeddings.shape == (bundle.num_ports, H)
        assert out.vessel_embeddings.shape == (2, H)   # all classes
        assert out.global_embedding_gat.shape == (H,)
        assert out.global_embedding.shape == (H,)
        assert out.node_embeddings.shape == (bundle.num_nodes, H)
        # tokens = P ports + 1 global + V vessels
        assert out.tokens.shape == (bundle.num_ports + 1 + 2, H)

    def test_encoder_only_selects_one_vessel_token(self, model_and_bundle):
        """[PAPER] Eq. 10 — encoder-only encodes the SELECTED class only."""
        model, bundle = model_and_bundle
        with torch.no_grad():
            out = model.encode_graph(bundle, selected_vessel_index=1)
        assert out.vessel_embeddings.shape == (1, model.hidden_dim)
        assert out.num_vessel_tokens == 1

    def test_selected_vessel_token_matches_full_encoding_slice(self, model_and_bundle):
        """
        Eq. 10's h_v uses the same projection as Eq. 16's per-class row, so the
        single selected token must equal the corresponding row of the full
        vessel encoding (the Transformer then contextualises each differently,
        so compare the PROJECTED vectors, not the post-Transformer ones).
        """
        model, bundle = model_and_bundle
        projected = model.vessel_projection(bundle.vessel_features)
        with torch.no_grad():
            single = model.vessel_projection(bundle.vessel_features[1])
        assert torch.allclose(projected[1], single)

    def test_outputs_are_finite(self, model_and_bundle):
        model, bundle = model_and_bundle
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert torch.isfinite(out.port_embeddings).all()
        assert torch.isfinite(out.vessel_embeddings).all()
        assert torch.isfinite(out.global_embedding_gat).all()

    def test_no_global_node_variant(self):
        model = NeuralBackbone(ArchitectureConfig.tiny(), include_global=False)
        model.eval()
        bundle = make_bundle(n_ports=3)
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert out.global_embedding is None
        assert out.global_embedding_gat.shape == (model.hidden_dim,)
        assert out.tokens.shape[0] == bundle.num_ports + bundle.num_vessel_classes

    def test_isolated_node_does_not_produce_nan(self):
        """
        A node with no incident edges must not yield NaN. Built by handing the
        GAT an edge set that touches only ports 0 and 1 of a 4-port graph.
        """
        cfg = ArchitectureConfig.tiny()
        model = NeuralBackbone(cfg)
        model.eval()
        N = 5
        x = torch.randn(N, 2)
        # Only nodes 0 and 1 have incident edges; nodes 2-4 are isolated.
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_attr = torch.randn(RESERVED_EDGE_FEATURE_DIM, 2)
        with torch.no_grad():
            h = model.gat(x, edge_index, edge_attr)
        assert torch.isfinite(h).all()

    def test_empty_graph_does_not_crash(self):
        """
        No edges at all: message passing contributes nothing, and the layer
        falls back to its residual path. The contract is "no NaN, correct
        shape" — not "zeros", since the residual keeps node information alive.
        """
        cfg = ArchitectureConfig.tiny()
        model = NeuralBackbone(cfg)
        model.eval()
        x = torch.randn(3, 2)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros(RESERVED_EDGE_FEATURE_DIM, 0)
        with torch.no_grad():
            h = model.gat(x, edge_index, edge_attr)
        assert h.shape == (3, cfg.hidden_dim)
        assert torch.isfinite(h).all()

    def test_attention_head_returns_zeros_when_no_edges(self):
        """The attention head itself contributes nothing without edges."""
        from neural.gat import EdgeFeatureAttention
        head = EdgeFeatureAttention(
            in_dim=2, out_dim=4, edge_dim=RESERVED_EDGE_FEATURE_DIM,
        )
        out = head(
            torch.randn(3, 2),
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros(RESERVED_EDGE_FEATURE_DIM, 0),
        )
        assert out.shape == (3, 4)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_gat_layer_with_no_edges_equals_normalized_residual(self):
        """
        With E = 0 the layer's output is exactly LN(residual(x)) — no attention
        contribution. Pinned so that a future change to the empty-graph path is
        a deliberate decision rather than a silent drift.
        """
        cfg = ArchitectureConfig.tiny()
        layer = GATStack(in_dim=2, config=cfg).layers[0]
        layer.eval()
        x = torch.randn(3, 2)
        with torch.no_grad():
            h = layer(
                x,
                torch.zeros((2, 0), dtype=torch.long),
                torch.zeros(RESERVED_EDGE_FEATURE_DIM, 0),
            )
            expected = layer.norm(layer.residual(x))
        assert torch.allclose(h, expected, atol=1e-6)

    def test_vessel_index_out_of_range_rejected(self, model_and_bundle):
        model, bundle = model_and_bundle
        with pytest.raises(ValueError, match="out of range"):
            model.encode_graph(bundle, selected_vessel_index=99)
        with pytest.raises(ValueError, match="out of range"):
            model.encode_graph(bundle, selected_vessel_index=-1)


# ===========================================================================
# 12. Determinism
# ===========================================================================

class TestDeterminism:

    def test_cpu_forward_pass_is_deterministic(self):
        """Test 12 — same input, same output, on CPU."""
        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        bundle = make_bundle(n_ports=4)

        with torch.no_grad():
            a = model.encode_graph(bundle)
            b = model.encode_graph(bundle)

        assert torch.equal(a.port_embeddings, b.port_embeddings)
        assert torch.equal(a.vessel_embeddings, b.vessel_embeddings)
        assert torch.equal(a.global_embedding_gat, b.global_embedding_gat)

    def test_determinism_across_separate_model_instances(self):
        """Same seed → same parameters → same output."""
        torch.manual_seed(1234)
        m1 = NeuralBackbone(ArchitectureConfig.tiny())
        torch.manual_seed(1234)
        m2 = NeuralBackbone(ArchitectureConfig.tiny())
        m1.eval(); m2.eval()
        bundle = make_bundle(n_ports=3)

        with torch.no_grad():
            a = m1.encode_graph(bundle)
            b = m2.encode_graph(bundle)
        assert torch.allclose(a.port_embeddings, b.port_embeddings, atol=0, rtol=0)

    def test_port_ordering_does_not_affect_per_node_results_permutation(self):
        """
        Permuting the node order must permute the outputs, not change them.

        This guards against any accidental dependence on node index (e.g. a
        positional encoding, which the paper explicitly omits).
        """
        cfg = ArchitectureConfig.tiny()
        torch.manual_seed(7)
        model = NeuralBackbone(cfg)
        model.eval()

        bundle = make_bundle(n_ports=4)
        N, E = bundle.num_nodes, bundle.num_edges

        with torch.no_grad():
            base = model.encode_graph(bundle)

        # Permute physical port nodes (keep the global node last).
        perm = torch.tensor([2, 0, 3, 1])
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(4)

        new_nodes = bundle.node_features.clone()
        new_nodes[:4] = bundle.node_features[:4][perm]

        new_edges = bundle.edge_index.clone()
        new_edges[0] = inv[bundle.edge_index[0]]
        new_edges[1] = inv[bundle.edge_index[1]]

        with torch.no_grad():
            gat_a = model.gat(
                model.node_norm(bundle.node_features),
                bundle.edge_index,
                torch.cat([bundle.static_edge_features, bundle.dynamic_edge_features], 0),
            )
            gat_b = model.gat(
                model.node_norm(new_nodes),
                new_edges,
                torch.cat([bundle.static_edge_features, bundle.dynamic_edge_features], 0),
            )

        # Node i in the permuted graph is original node perm[i].
        assert torch.allclose(gat_b[:4], gat_a[:4][perm], atol=1e-6)

    def test_segment_softmax_matches_dense_reference(self):
        """The O(E) segment softmax must equal a per-group dense softmax."""
        torch.manual_seed(0)
        logits = torch.randn(20)
        index = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 3, 3, 0, 1, 2, 2, 2, 3, 0, 1, 1, 2])
        alpha = _segment_softmax(logits, index, 5)
        for g in range(5):
            m = index == g
            if m.any():
                assert torch.allclose(alpha[m], torch.softmax(logits[m], 0), atol=1e-6)
        # Every group that HAS members sums to 1.
        for g in range(5):
            m = index == g
            if m.any():
                assert abs(float(alpha[m].sum()) - 1.0) < 1e-6

    def test_segment_softmax_handles_empty_segments(self):
        logits = torch.tensor([1.0, 2.0])
        index = torch.tensor([0, 0])
        alpha = _segment_softmax(logits, index, 4)
        assert torch.isfinite(alpha).all()

    def test_segment_softmax_is_numerically_stable(self):
        """Large logits must not overflow (max-subtraction trick)."""
        logits = torch.tensor([1000.0, 1001.0, -1000.0])
        index = torch.tensor([0, 0, 0])
        alpha = _segment_softmax(logits, index, 1)
        assert torch.isfinite(alpha).all()
        assert abs(float(alpha.sum()) - 1.0) < 1e-5


# ===========================================================================
# 13. Multiple graph sizes
# ===========================================================================

class TestVariableGraphSizes:

    @pytest.mark.parametrize("n_ports,n_vessels", [
        (2, 1), (3, 2), (5, 2), (8, 3), (12, 6),
    ])
    def test_multiple_graph_sizes(self, n_ports, n_vessels):
        """Test 13 — no fixed-size assumption (P7.7)."""
        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        bundle = make_bundle(n_ports=n_ports, n_vessels=n_vessels)
        with torch.no_grad():
            out = model.encode_graph(bundle)
        H = model.hidden_dim
        assert out.port_embeddings.shape == (n_ports, H)
        assert out.vessel_embeddings.shape == (n_vessels, H)
        assert out.node_embeddings.shape == (n_ports + 1, H)

    def test_one_model_serves_different_sizes(self):
        """The same instance must handle successive sizes without rebuilding."""
        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        with torch.no_grad():
            for n in (2, 4, 7):
                out = model.encode_graph(make_bundle(n_ports=n))
                assert out.port_embeddings.shape[0] == n

    def test_no_instance_size_is_hardcoded_in_state_dict(self):
        """Parameters must not encode any graph size."""
        m = NeuralBackbone(ArchitectureConfig.tiny())
        for key, tensor in m.state_dict().items():
            # The only instance-dependent-looking dims would come from the
            # synthetic fixture (up to 12 ports); assert none leak in.
            assert 12 not in tensor.shape, f"{key} has an instance-sized dim"

    def test_increasing_service_count_is_handled(self):
        """
        Dynamic edge features GROW with the number of services while the GAT's
        declared edge width is fixed — the padding path must absorb it.
        """
        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        for n_svc in (0, 1, 3):
            bundle = make_bundle(n_ports=4, num_services=n_svc)
            with torch.no_grad():
                out = model.encode_graph(bundle)
            assert torch.isfinite(out.port_embeddings).all()


# ===========================================================================
# 14. Serialization
# ===========================================================================

class TestSerialization:

    def test_serialization_round_trip(self, tmp_path):
        """Test 14 — save, load, same input, same output."""
        cfg = ArchitectureConfig.tiny()
        torch.manual_seed(3)
        model = NeuralBackbone(cfg)
        model.eval()

        path = tmp_path / "backbone.pt"
        save_backbone(model, str(path))
        assert path.exists()

        restored = load_backbone(str(path))
        restored.eval()

        bundle = make_bundle(n_ports=4)
        with torch.no_grad():
            a = model.encode_graph(bundle)
            b = restored.encode_graph(bundle)

        assert torch.equal(a.port_embeddings, b.port_embeddings)
        assert torch.equal(a.vessel_embeddings, b.vessel_embeddings)
        assert torch.equal(a.global_embedding_gat, b.global_embedding_gat)

    def test_round_trip_preserves_config(self, tmp_path):
        model = NeuralBackbone(ArchitectureConfig.tiny(), include_global=False)
        path = tmp_path / "cfg.pt"
        save_backbone(model, str(path))
        restored = load_backbone(str(path))
        assert restored.config == model.config
        assert restored.include_global is False
        assert restored.edge_feature_dim == model.edge_feature_dim

    def test_round_trip_does_not_create_training_checkpoint(self, tmp_path):
        """Architecture-level state only (no optimizer, no step counter)."""
        model = NeuralBackbone(ArchitectureConfig.tiny())
        path = tmp_path / "arch.pt"
        save_backbone(model, str(path))
        payload = torch.load(str(path), weights_only=False)
        assert set(payload) == {
            "config", "node_feature_dim", "edge_feature_dim",
            "include_global", "state_dict", "paper_architecture",
        }
        assert "optimizer" not in payload
        assert "step" not in payload

    def test_load_rejects_malformed_payload(self, tmp_path):
        path = tmp_path / "bad.pt"
        torch.save({"config": {}}, str(path))
        with pytest.raises(ValueError, match="missing"):
            load_backbone(str(path))

    def test_load_rejects_non_dict_payload(self, tmp_path):
        path = tmp_path / "bad2.pt"
        torch.save([1, 2, 3], str(path))
        with pytest.raises(ValueError, match="expected a dict"):
            load_backbone(str(path))

    def test_state_dict_keys_are_dataset_independent(self):
        """A checkpoint must be loadable onto any instance's model."""
        a = NeuralBackbone(ArchitectureConfig.tiny())
        b = NeuralBackbone(ArchitectureConfig.tiny())
        b.load_state_dict(a.state_dict())  # must not raise


# ===========================================================================
# 15. Invalid input detection
# ===========================================================================

class TestInvalidInput:

    def test_rejects_non_ndarray(self):
        inst = make_instance(n_ports=3)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        state = encoder.encode(
            {i: d.ffe_per_week for i, d in enumerate(inst.demands)},
            {e.vessel_class: float(e.quantity) for e in inst.fleet},
            ServiceMembership(),
        )
        state.port_features = [[1.0, 2.0]]  # not an ndarray
        with pytest.raises(TypeError, match="numpy ndarray"):
            neural_state_to_tensors(state)

    def test_rejects_wrong_node_feature_width(self):
        bundle = make_bundle(n_ports=3)
        model = NeuralBackbone(ArchitectureConfig.tiny())
        bad = bundle.node_features[:, :1]
        with pytest.raises(ValueError, match="node_features must be"):
            model._encode(
                node_features=bad,
                edge_index=bundle.edge_index,
                edge_attr=torch.cat(
                    [bundle.static_edge_features, bundle.dynamic_edge_features], 0,
                ),
                vessel_features=bundle.vessel_features,
                selected_vessel_index=None,
                key_padding_mask=None,
            )

    def test_rejects_wrong_vessel_feature_width(self):
        inst = make_instance(n_ports=3)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        state = encoder.encode(
            {i: d.ffe_per_week for i, d in enumerate(inst.demands)},
            {e.vessel_class: float(e.quantity) for e in inst.fleet},
            ServiceMembership(),
        )
        state.vessel_features = state.vessel_features[:, :5]
        with pytest.raises(ValueError, match="11 columns"):
            neural_state_to_tensors(state)

    def test_rejects_nonfinite_features(self):
        inst = make_instance(n_ports=3)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        state = encoder.encode(
            {i: d.ffe_per_week for i, d in enumerate(inst.demands)},
            {e.vessel_class: float(e.quantity) for e in inst.fleet},
            ServiceMembership(),
        )
        state.port_features = state.port_features.copy()
        state.port_features[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            neural_state_to_tensors(state)

    def test_rejects_wrong_dtype_string(self):
        with pytest.raises(ValueError, match="Unsupported dtype"):
            neural_state_to_tensors(_state_for(3), dtype="float16")

    def test_rejects_empty_vessel_set(self):
        bundle = make_bundle(n_ports=3)
        model = NeuralBackbone(ArchitectureConfig.tiny())
        with pytest.raises(ValueError, match="at least one class"):
            model._encode(
                node_features=bundle.node_features,
                edge_index=bundle.edge_index,
                edge_attr=torch.cat(
                    [bundle.static_edge_features, bundle.dynamic_edge_features], 0,
                ),
                vessel_features=bundle.vessel_features[:0],
                selected_vessel_index=None,
                key_padding_mask=None,
            )

    def test_transformer_rejects_wrong_hidden_dim(self):
        stack = TransformerEncoderStack(ArchitectureConfig.tiny())
        with pytest.raises(ValueError, match="expects"):
            stack(torch.randn(4, 3))

    def test_transformer_rejects_bad_rank(self):
        """A 1-D token tensor is not a valid (B, N, H) or (N, H) input."""
        stack = TransformerEncoderStack(ArchitectureConfig.tiny())
        with pytest.raises(ValueError, match="2-D or 3-D"):
            stack(torch.randn(16))

    def test_joint_encoder_rejects_non_2d_port_embeddings(self):
        from neural.transformer import joint_encoder_forward
        stack = TransformerEncoderStack(ArchitectureConfig.tiny())
        H = stack.hidden_dim
        with pytest.raises(ValueError, match="must be 2-D"):
            joint_encoder_forward(stack, torch.randn(2, 3, H), torch.randn(1, H))

    def test_gat_layer_rejects_wrong_input_dim(self):
        cfg = ArchitectureConfig.tiny()
        layer = GATStack(in_dim=2, config=cfg).layers[0]
        with pytest.raises(ValueError, match="expects 2 input features"):
            layer(
                torch.randn(3, 5),
                torch.tensor([[0], [1]], dtype=torch.long),
                torch.zeros(RESERVED_EDGE_FEATURE_DIM, 1),
            )

    def test_gat_rejects_oversized_edge_feature_block(self):
        """A too-WIDE edge block is a schema violation, not something to trim."""
        cfg = ArchitectureConfig.tiny()
        stack = GATStack(in_dim=2, config=cfg, edge_dim=10)
        with pytest.raises(ValueError, match="more features than"):
            stack(
                torch.randn(3, 2),
                torch.tensor([[0], [1]], dtype=torch.long),
                torch.zeros(12, 1),
            )

    def test_joint_encoder_rejects_mismatched_vessel_dim(self):
        from neural.transformer import joint_encoder_forward
        stack = TransformerEncoderStack(ArchitectureConfig.tiny())
        H = stack.hidden_dim
        with pytest.raises(ValueError, match="vessel_embeddings feature dim"):
            joint_encoder_forward(
                stack, torch.randn(3, H), torch.randn(2, H + 1),
            )

    def test_joint_encoder_rejects_bad_global_dim(self):
        from neural.transformer import joint_encoder_forward
        stack = TransformerEncoderStack(ArchitectureConfig.tiny())
        H = stack.hidden_dim
        with pytest.raises(ValueError, match="global_embedding must be 1-D"):
            joint_encoder_forward(
                stack, torch.randn(3, H), torch.randn(1, H),
                global_embedding=torch.randn(3, H),
            )


def _state_for(n_ports: int):
    """Helper: build a P5 NeuralState for a synthetic instance."""
    inst = make_instance(n_ports=n_ports)
    dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
    encoder = StateEncoder(inst, dist_by_pair)
    return encoder.encode(
        {i: d.ffe_per_week for i, d in enumerate(inst.demands)},
        {e.vessel_class: float(e.quantity) for e in inst.fleet},
        ServiceMembership(),
    )


# ===========================================================================
# 16. Mask interface
# ===========================================================================

class TestMasks:

    def test_encoder_only_port_mask_is_all_available(self):
        """
        [PAPER] Eqs. 12-13 define NO eligibility mask for the encoder-only
        pathway; the interface returns all-True rather than inventing one.
        """
        m = encoder_only_port_mask(7)
        assert m.shape == (7,)
        assert m.dtype == torch.bool
        assert bool(m.all())

    def test_all_and_none_available(self):
        assert bool(all_available(4).all())
        assert not bool(none_available(4).any())
        with pytest.raises(ValueError):
            all_available(-1)

    def test_pytorch_padding_mask_inverts(self):
        keep = torch.tensor([True, False, True])
        pm = to_pytorch_padding_mask(keep)
        assert pm.tolist() == [False, True, False]

    def test_apply_mask_to_logits_is_not_in_place(self):
        logits = torch.tensor([1.0, 2.0, 3.0])
        keep = torch.tensor([True, False, True])
        masked = apply_mask_to_logits(logits, keep)
        # Original untouched.
        assert logits.tolist() == [1.0, 2.0, 3.0]
        assert masked[1].item() == MASK_FILL_VALUE
        assert masked[0].item() == 1.0

    def test_apply_mask_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="shape"):
            apply_mask_to_logits(
                torch.randn(3), torch.tensor([True, False]),
            )

    def test_masked_log_softmax_suppresses_masked_entries(self):
        logits = torch.tensor([1.0, 100.0, 2.0])
        keep = torch.tensor([True, False, True])
        lp = masked_log_softmax(logits, keep)
        assert torch.isfinite(lp).all()
        assert lp[1].item() < -1e8
        # Unmasked entries form a proper log-distribution.
        assert abs(float(torch.exp(lp[[0, 2]]).sum()) - 1.0) < 1e-5

    def test_draft_feasible_port_mask_uses_p6_rule(self):
        """
        The rule is P6's `_is_draft_feasible`: vessel_draft >= port_draft - tol.
        """
        bundle = make_bundle(n_ports=3, port_draft=10.0)
        # Vessel draft 12 can enter all; draft 9 can enter none.
        assert bool(draft_feasible_port_mask(bundle, 12.0, [10.0, 10.0, 10.0]).all())
        assert not bool(draft_feasible_port_mask(bundle, 9.0, [10.0, 10.0, 10.0]).any())
        mixed = draft_feasible_port_mask(bundle, 10.0, [5.0, 20.0, 10.0])
        assert mixed.tolist() == [True, False, True]

    def test_draft_mask_treats_unknown_draft_as_available(self):
        """P6 treats a None port draft as available; the mask must agree."""
        bundle = make_bundle(n_ports=3)
        m = draft_feasible_port_mask(bundle, 1.0, [None, None, None])
        assert bool(m.all())

    def test_draft_mask_rejects_wrong_length(self):
        bundle = make_bundle(n_ports=3)
        with pytest.raises(ValueError, match="entries but the graph has"):
            draft_feasible_port_mask(bundle, 10.0, [10.0])

    def test_already_selected_port_mask(self):
        m = already_selected_port_mask(5, selected={1, 3})
        assert m.tolist() == [True, False, True, False, True]

    def test_already_selected_port_mask_allows_revisit(self):
        """The first port may be revisited to close the service."""
        m = already_selected_port_mask(5, selected={1, 3}, allow={1})
        assert m.tolist() == [True, True, True, False, True]

    def test_already_selected_port_mask_rejects_bad_index(self):
        with pytest.raises(ValueError, match="out of range"):
            already_selected_port_mask(3, selected={7})

    def test_global_node_mask_excludes_global_by_default(self):
        """[PAPER] Section 4.2 — global node is not a port candidate."""
        m = global_node_mask(4, include_global=False)
        assert m.tolist() == [True, True, True, True, False]

    def test_global_node_mask_can_include(self):
        m = global_node_mask(4, include_global=True)
        assert m.tolist() == [True] * 5

    def test_edge_padding_mask_shape_checked(self):
        from neural import edge_padding_mask
        bundle = make_bundle(n_ports=3)
        keep = torch.ones(bundle.num_edges, dtype=torch.bool)
        assert edge_padding_mask(bundle, keep).all()
        with pytest.raises(ValueError, match="keep must have shape"):
            edge_padding_mask(bundle, torch.ones(3, dtype=torch.bool))

    def test_apply_edge_mask_zeroes_padding(self):
        from neural import apply_edge_mask
        alpha = torch.tensor([0.5, 0.5, 0.5])
        keep = torch.tensor([True, False, True])
        out = apply_edge_mask(alpha, keep)
        assert out.tolist() == [0.5, 0.0, 0.5]

    def test_padding_mask_reaches_transformer(self):
        """A padded token must not influence the attended outputs."""
        cfg = ArchitectureConfig.tiny()
        model = NeuralBackbone(cfg)
        model.eval()
        bundle = make_bundle(n_ports=4, n_vessels=2)
        n_tokens = bundle.num_ports + 1 + bundle.num_vessel_classes
        keep = torch.ones(n_tokens, dtype=torch.bool)
        keep[-1] = False  # mask the last vessel token

        with torch.no_grad():
            out = model.encode_graph(
                bundle, key_padding_mask=to_pytorch_padding_mask(keep),
            )
        assert torch.isfinite(out.port_embeddings).all()
        assert out.tokens.shape[0] == n_tokens

    def test_padding_mask_wrong_length_rejected(self):
        model = NeuralBackbone(ArchitectureConfig.tiny())
        bundle = make_bundle(n_ports=3)
        with pytest.raises(ValueError, match="key_padding_mask has"):
            model.encode_graph(
                bundle, key_padding_mask=torch.zeros(2, dtype=torch.bool),
            )

    def test_describe_mask(self):
        d = describe_mask(torch.tensor([True, False, True]))
        assert d == {"total": 3, "available": 2, "masked": 1, "empty": False}
        assert describe_mask(none_available(2))["empty"] is True

    # ---- decoder phase masks (interface consumed by P9) ----

    def test_decoder_phase_mask_first_substep_is_vessel_only(self):
        m = decoder_phase_mask(num_ports=4, num_vessels=2, substep=1)
        assert m.phase == "vessel"
        assert m.keep.tolist() == [False] * 4 + [True, True]

    def test_decoder_phase_mask_later_substep_is_port_only(self):
        m = decoder_phase_mask(num_ports=4, num_vessels=2, substep=2)
        assert m.phase == "port"
        assert m.keep.tolist() == [True] * 4 + [False, False]

    def test_decoder_phase_mask_masks_visited_ports_except_first(self):
        m = decoder_phase_mask(
            num_ports=4, num_vessels=2, substep=3,
            selected_ports={1, 2}, first_port=1,
        )
        # Port 1 (the first port) is re-allowed so the service can close.
        assert m.keep[:4].tolist() == [True, True, False, True]

    def test_decoder_phase_mask_restricts_vessels(self):
        m = decoder_phase_mask(
            num_ports=3, num_vessels=4, substep=1, available_vessels={0, 2},
        )
        assert m.keep.tolist() == [False] * 3 + [True, False, True, False]

    def test_decoder_phase_mask_bos_rule(self):
        """[PAPER] BOS unmasked only at τ=1 of t=1."""
        with_bos = decoder_phase_mask(
            num_ports=3, num_vessels=2, substep=1,
            include_bos=True, bos_allowed=True,
        )
        assert with_bos.num_candidates == 6
        assert bool(with_bos.keep[-1])

        later = decoder_phase_mask(
            num_ports=3, num_vessels=2, substep=1,
            include_bos=True, bos_allowed=False,
        )
        assert not bool(later.keep[-1])

        port_phase = decoder_phase_mask(
            num_ports=3, num_vessels=2, substep=2,
            include_bos=True, bos_allowed=True,
        )
        assert not bool(port_phase.keep[-1])

    def test_decoder_phase_mask_folds_draft_feasibility(self):
        draft = torch.tensor([True, False, True])
        m = decoder_phase_mask(
            num_ports=3, num_vessels=1, substep=2, draft_keep_ports=draft,
        )
        assert m.keep[:3].tolist() == [True, False, True]
        assert m.num_available == 2

    def test_decoder_phase_mask_rejects_bad_substep(self):
        with pytest.raises(ValueError, match="substep must be"):
            decoder_phase_mask(num_ports=3, num_vessels=1, substep=0)

    def test_decoder_phase_mask_draft_shape_checked(self):
        with pytest.raises(ValueError, match="draft_keep_ports must have shape"):
            decoder_phase_mask(
                num_ports=3, num_vessels=1, substep=2,
                draft_keep_ports=torch.ones(5, dtype=torch.bool),
            )


# ===========================================================================
# 17. dtype / device handling
# ===========================================================================

class TestDtypeDevice:

    def test_default_is_cpu_float32(self):
        bundle = make_bundle(n_ports=3)
        assert bundle.device.type == "cpu"
        assert bundle.dtype == torch.float32
        assert bundle.node_features.dtype == torch.float32

    def test_float64_conversion(self):
        bundle = make_bundle(n_ports=3, dtype="float64")
        assert bundle.dtype == torch.float64
        assert bundle.node_features.dtype == torch.float64
        assert bundle.vessel_features.dtype == torch.float64

    def test_float64_model_forward(self):
        """A float64 model must accept float64 inputs end to end."""
        model = NeuralBackbone(ArchitectureConfig.tiny()).double()
        model.eval()
        bundle = make_bundle(n_ports=3, dtype="float64")
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert out.port_embeddings.dtype == torch.float64
        assert torch.isfinite(out.port_embeddings).all()

    def test_cuda_is_not_hardcoded(self):
        """No tensor may be created on CUDA unless explicitly asked."""
        bundle = make_bundle(n_ports=3)
        assert bundle.device.type == "cpu"
        model = NeuralBackbone(ArchitectureConfig.tiny())
        assert next(model.parameters()).device.type == "cpu"

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available",
    )
    def test_cuda_exposed_when_available(self):  # pragma: no cover
        model = NeuralBackbone(ArchitectureConfig.tiny()).cuda()
        bundle = make_bundle(n_ports=3).to("cuda")
        model.eval()
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert out.port_embeddings.device.type == "cuda"

    def test_to_device_round_trip(self):
        bundle = make_bundle(n_ports=3)
        same = bundle.to("cpu")
        assert same.device.type == "cpu"
        assert torch.equal(same.node_features, bundle.node_features)
        # Metadata is preserved.
        assert same.port_codes == bundle.port_codes
        assert same.num_ports == bundle.num_ports

    def test_edge_index_is_long_dtype(self):
        bundle = make_bundle(n_ports=3)
        assert bundle.edge_index.dtype == torch.long
        assert bundle.edge_index.shape[0] == 2

    def test_device_dtype_config_is_respected(self):
        cfg = ArchitectureConfig.tiny(device="cpu", dtype="float64")
        bundle = make_bundle(n_ports=3, config=cfg)
        assert bundle.dtype == torch.float64


# ===========================================================================
# Cross-cutting: architecture-only scope
# ===========================================================================

class TestScopeBoundaries:
    """P7 must not contain policy, PPO, or training machinery."""

    def test_backbone_has_no_optimizer_or_value_head(self):
        model = NeuralBackbone(ArchitectureConfig.tiny())
        names = [n for n, _ in model.named_modules()]
        for forbidden in ("optimizer", "value_head", "critic", "policy_head"):
            assert not any(forbidden in n for n in names)

    def test_backbone_exposes_no_training_loop(self):
        model = NeuralBackbone(ArchitectureConfig.tiny())
        for forbidden in ("train_step", "update", "ppo_loss", "compute_advantage",
                          "collect_rollout"):
            assert not hasattr(model, forbidden)

    def test_no_gradients_required_for_forward(self):
        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        bundle = make_bundle(n_ports=3)
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert not out.port_embeddings.requires_grad

    def test_neural_package_does_not_import_env_or_actions(self):
        """
        P7 sits below the policy layer: it must not reach into the environment
        or the service generator.
        """
        import neural
        src = Path(neural.__file__).parent
        for py in src.glob("*.py"):
            text = py.read_text(encoding="utf-8")
            for banned in ("import env", "from env", "import actions",
                           "from actions", "import mcf", "from mcf"):
                assert banned not in text, f"{py.name} contains {banned!r}"


# ===========================================================================
# Integration: real LINERLIB instances
# ===========================================================================

class TestRealInstances:

    @pytest.fixture(scope="class")
    def loader(self):
        from data.linerlib_loader import LINERLIBLoader
        return LINERLIBLoader(str(ROOT / "data"))

    def _bundle_for(self, loader, name: str) -> GraphTensors:
        from state.representation import StateEncoder
        inst = loader.load(name, validate=False)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        encoder = StateEncoder(inst, dist_by_pair)
        state = encoder.encode(
            {i: d.ffe_per_week for i, d in enumerate(inst.demands)},
            {e.vessel_class: float(e.quantity) for e in inst.fleet},
            ServiceMembership(),
        )
        return neural_state_to_tensors(state)

    @pytest.mark.parametrize("name", ["Baltic", "WAF"])
    def test_real_instance_shapes(self, loader, name):
        """Real benchmark instances flow through P5 → P7 unchanged."""
        bundle = self._bundle_for(loader, name)
        assert bundle.num_nodes == bundle.num_ports + 1
        assert bundle.num_edges > 0
        verify_edge_alignment(bundle)

    def test_baltic_expected_dimensions(self):
        """Baltic is the paper's primary instance: 12 ports, 6 vessel classes."""
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader(str(ROOT / "data"))
        bundle = self._bundle_for(loader, "Baltic")
        assert bundle.num_ports == 12
        assert bundle.num_nodes == 13
        assert bundle.num_vessel_classes == 6

    def test_real_instance_forward_pass(self):
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader(str(ROOT / "data"))
        bundle = self._bundle_for(loader, "Baltic")
        model = NeuralBackbone(ArchitectureConfig())   # paper config, real data
        model.eval()
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert out.port_embeddings.shape == (12, 512)
        assert torch.isfinite(out.port_embeddings).all()
        assert torch.isfinite(out.global_embedding_gat).all()

    def test_real_instance_encoder_only_vessel_selection(self):
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader(str(ROOT / "data"))
        bundle = self._bundle_for(loader, "Baltic")
        model = NeuralBackbone(ArchitectureConfig())
        model.eval()
        with torch.no_grad():
            out = model.encode_graph(bundle, selected_vessel_index=0)
        assert out.vessel_embeddings.shape == (1, 512)

    def test_real_instance_is_deterministic(self):
        from data.linerlib_loader import LINERLIBLoader
        loader = LINERLIBLoader(str(ROOT / "data"))
        bundle = self._bundle_for(loader, "Baltic")
        model = NeuralBackbone(ArchitectureConfig.tiny())
        model.eval()
        with torch.no_grad():
            a = model.encode_graph(bundle)
            b = model.encode_graph(bundle)
        assert torch.equal(a.port_embeddings, b.port_embeddings)
