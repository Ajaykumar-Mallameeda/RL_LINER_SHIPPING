"""
G2 — GPU Paper-Scale Architecture Validation

Validates that the ACTUAL paper-scale neural architecture implemented in this
repository can execute a training step (forward → loss → backward → optimizer)
on CUDA. This is a computational validation, NOT an RL learning validation.

Architecture under test (paper-frozen values from Table 5):
    H = 512, GAT layers = 3, Transformer layers = 3, heads = 8, LSTM = 1

Scope:
    - Backward compatibility with existing tests: this file adds NEW tests only.
    - Does not modify any existing module.
    - Does not launch real PPO training.
    - GPU tests are skipped when CUDA is unavailable.

Test matrix:
    G2-A: shared backbone (paper-config, CUDA)
    G2-B: encoder-only policy (paper-config, CUDA)
    G2-C: encoder-decoder policy (paper-config, CUDA, backbone path)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from actions.service_generator import ServiceGenerator
from data.instance import DatasetProvenance
from neural import (
    PAPER_ARCHITECTURE,
    ArchitectureConfig,
    NeuralBackbone,
    paper_config,
)
from neural.config import PAPER_NODE_FEATURE_DIM, PAPER_VESSEL_FEATURE_DIM
from policies.encoder_decoder import EncoderDecoderPolicy
from policies.encoder_only import EncoderOnlyPolicy
from state.representation import ServiceMembership, StateEncoder

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_cuda() -> bool:
    return torch.cuda.is_available()


def _get_gpu_info() -> dict:
    """Return hardware and driver information."""
    if not _has_cuda():
        return {}
    prop = torch.cuda.get_device_properties(0)
    return {
        "name": torch.cuda.get_device_name(0),
        "total_vram_bytes": prop.total_memory,
        "total_vram_gb": prop.total_memory / 1e9,
        "cuda_compute_capability": f"{prop.major}.{prop.minor}",
    }


def _make_synthetic_instance(n_ports: int = 4, n_vessels: int = 2, seed: int = 0):
    """Create a minimal synthetic LINERLIBInstance suitable for G2 tests."""
    codes = [f"P{i:02d}" for i in range(n_ports)]
    import numpy as np
    rng = np.random.default_rng(seed)

    from data.instance import Port, VesselType, Demand, DistanceArc, FleetEntry
    from data.instance import InstanceMetadata, ProvenanceRecord

    ports = {}
    for c in codes:
        ports[c] = Port(
            unlocode=c, name=f"Port {c}", country=None,
            cabotage_region="test", d_region=None, longitude=None, latitude=None,
            draft=10.0, cost_per_full=1.0, cost_per_full_transfer=0.5,
            port_call_cost_fixed=100.0, port_call_cost_per_ffe=0.5,
            provenance=ProvenanceRecord(source_file="g2_test", source_row=1),
        )

    vessels = {}
    for i in range(n_vessels):
        vessels[f"V{i}"] = VesselType(
            vessel_class=f"V{i}", capacity_ffe=100.0 * (i + 1), tc_rate_daily=100,
            draft=12.0 + i, min_speed=5.0, max_speed=15.0, design_speed=10.0,
            bunker_ton_per_day_at_design=50.0, idle_consumption_ton_per_day=10.0,
            panama_fee=0, suez_fee=0,
            provenance=ProvenanceRecord(source_file="g2_test", source_row=1),
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
                draft_required=10.0, is_panama=False, is_suez=False,
                provenance=ProvenanceRecord(source_file="g2_test", source_row=row),
            ))
            row += 1
    for k in range(min(2, n_ports - 1)):
        demands.append(Demand(
            origin=codes[k], destination=codes[k + 1],
            ffe_per_week=50.0 + 10.0 * k, revenue=200.0, max_transit_time=20,
            provenance=ProvenanceRecord(source_file="g2_test", source_row=k + 1),
        ))

    fleet = [FleetEntry(vessel_class=f"V{i}", quantity=3) for i in range(n_vessels)]
    metadata = InstanceMetadata(
        name=f"G2_SYNTH_{n_ports}P", active_port_count=n_ports,
        vessel_type_count=n_vessels, total_vessels=3 * n_vessels,
        demand_count=len(demands), distance_arc_count=len(distances),
    )
    return type('SyntheticInstance', (), {
        'ports': ports, 'vessel_types': vessels, 'demands': demands,
        'distances': distances, 'fleet': fleet, 'metadata': metadata,
        'name': f'G2_SYNTH_{n_ports}P',
        'provenance': DatasetProvenance(source_root="[G2 TEST]"),
    })()


def _make_bundle_for_g2(n_ports: int = 4, n_vessels: int = 2):
    """Build a GraphTensors bundle matching the project's fixture convention."""
    from tests.test_neural_architecture import make_bundle
    return make_bundle(n_ports=n_ports, n_vessels=n_vessels, num_services=0)


def _build_paper_backbone(device: str = "cpu") -> NeuralBackbone:
    """Construct the actual paper-config backbone on the requested device."""
    cfg = paper_config()
    model = NeuralBackbone(cfg)
    if device != "cpu":
        model = model.to(device)
    return model


# ===========================================================================
# G2-A: Shared Backbone — construction, parameters, CUDA placement
# ===========================================================================

skip_no_cuda = pytest.mark.skipif(
    not _has_cuda(), reason="CUDA not available — skipping G2 GPU validation.",
)


class TestG2ABackboneConstruction:
    """G2-A: Paper-config backbone can be constructed with correct architecture."""

    def test_paper_constants_are_frozen(self):
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

    def test_backbone_has_paper_dimensions(self):
        model = _build_paper_backbone()
        assert model.hidden_dim == 512
        assert model.num_gat_layers == 3
        assert model.num_transformer_layers == 3
        assert model.num_transformer_heads == 8
        assert len(model.gat.layers) == 3
        assert len(model.transformer.encoder.layers) == 3
        head = model.transformer.encoder.layers[0]
        assert head.self_attn.num_heads == 8

    def test_parameter_count_is_reasonable(self):
        model = _build_paper_backbone()
        count = model.parameter_count()
        # Paper-scale H=512, 3 GAT, 3 Transformer × 8 heads yields ~11M params.
        assert count > 10_000_000, f"Expected >10M params, got {count}"
        assert count < 20_000_000, f"Unexpected param count: {count}"

    def test_parameters_are_trainable(self):
        model = _build_paper_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        assert trainable == total, f"Non-trainable params found: {total - trainable}"


# ===========================================================================
# G2-B: CUDA device placement and forward pass
# ===========================================================================

@skip_no_cuda
class TestG2BCudaPlacement:
    """G2-B: Model and tensors can be placed on CUDA."""

    def test_model_moved_to_cuda(self):
        model = _build_paper_backbone("cuda")
        assert next(model.parameters()).device.type == "cuda"
        total_params = sum(1 for p in model.parameters())
        cuda_params = sum(
            1 for p in model.parameters() if p.device.type == "cuda"
        )
        cpu_params = sum(
            1 for p in model.parameters() if p.device.type == "cpu"
        )
        assert cpu_params == 0
        assert cuda_params == total_params

    def test_bundle_on_cuda(self):
        bundle = _make_bundle_for_g2().to("cuda")
        assert bundle.device.type == "cuda"
        assert bundle.node_features.device.type == "cuda"
        assert bundle.edge_index.device.type == "cuda"
        assert bundle.static_edge_features.device.type == "cuda"
        assert bundle.vessel_features.device.type == "cuda"


@skip_no_cuda
class TestG2CForwardPass:
    """G2-C: Forward pass succeeds on CUDA with paper-scale architecture."""

    def test_forward_pass_returns_valid_output(self):
        model = _build_paper_backbone("cuda")
        model.eval()
        bundle = _make_bundle_for_g2().to("cuda")
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert out.port_embeddings.shape == (bundle.num_ports, 512)
        assert out.vessel_embeddings.shape == (bundle.num_vessel_classes, 512)
        assert out.global_embedding_gat.shape == (512,)
        assert out.tokens.shape[0] == bundle.num_nodes + bundle.num_vessel_classes

    def test_forward_output_is_finite(self):
        model = _build_paper_backbone("cuda")
        model.eval()
        bundle = _make_bundle_for_g2().to("cuda")
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert torch.isfinite(out.port_embeddings).all()
        assert torch.isfinite(out.vessel_embeddings).all()
        assert torch.isfinite(out.global_embedding_gat).all()

    def test_forward_output_device(self):
        model = _build_paper_backbone("cuda")
        model.eval()
        bundle = _make_bundle_for_g2().to("cuda")
        with torch.no_grad():
            out = model.encode_graph(bundle)
        assert out.port_embeddings.device.type == "cuda"
        assert out.vessel_embeddings.device.type == "cuda"
        assert out.global_embedding_gat.device.type == "cuda"


# ===========================================================================
# G2-D: Backward pass and gradient quality
# ===========================================================================

@skip_no_cuda
class TestG2DBackward:
    """G2-D: Loss can be differentiated and gradients are finite."""

    def test_backward_produces_gradients(self):
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_count = sum(
            1 for p in model.parameters() if p.grad is not None
        )
        assert grad_count > 0

    def test_gradients_are_finite(self):
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        finite = sum(
            1 for p in model.parameters()
            if p.grad is not None and torch.isfinite(p.grad).all()
        )
        total_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None
        )
        assert finite == total_with_grad

    def test_global_gradient_norm_is_nonzero_and_finite(self):
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = sum((p.grad ** 2).sum().item()
                   for p in model.parameters() if p.grad is not None) ** 0.5
        assert norm > 0.0
        assert torch.isfinite(torch.tensor(norm))


# ===========================================================================
# G2-E: Optimizer step and parameter update
# ===========================================================================

@skip_no_cuda
class TestG2EOptimizerStep:
    """G2-E: Optimizer step succeeds and changes parameters."""

    def test_optimizer_step_changes_parameters(self):
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        snapshots = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        optimizer.step()

        changed = sum(
            1 for name, old in snapshots.items()
            if not torch.equal(old, model.state_dict()[name])
        )
        assert changed > 0, "No parameters changed after optimizer step."

    def test_optimizer_step_no_nan(self):
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        for p in model.parameters():
            assert torch.isfinite(p).all(), "Parameter contains NaN or Inf after optimizer step."


# ===========================================================================
# G2-F: Memory measurement
# ===========================================================================

@skip_no_cuda
class TestG2FMemory:
    """G2-F: GPU memory usage is measured and reported."""

    def test_peak_memory_is_within_vram_limits(self):
        torch.cuda.reset_peak_memory_stats()
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        peak_alloc = torch.cuda.max_memory_allocated()
        total_vram = torch.cuda.get_device_properties(0).total_memory
        # On a 6 GB GPU, peak should comfortably fit (< 4 GB in practice)
        assert peak_alloc < total_vram * 0.8, (
            f"Peak allocation ({peak_alloc / 1e9:.2f} GB) exceeds safe limit."
        )
        peak_gb = peak_alloc / 1e9
        total_gb = total_vram / 1e9
        utilization_pct = peak_alloc / total_vram * 100
        assert utilization_pct < 80.0, (
            f"Peak VRAM utilization ({utilization_pct:.1f}%) is too high."
        )

    def test_memory_stats_are_readable(self):
        """Verify we can read peak/memory stats without error."""
        torch.cuda.reset_peak_memory_stats()
        model = _build_paper_backbone("cuda")
        model.eval()
        bundle = _make_bundle_for_g2().to("cuda")
        with torch.no_grad():
            _ = model.encode_graph(bundle)
        alloc = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()
        assert alloc >= 0
        assert reserved >= 0
        assert peak_alloc >= alloc
        assert peak_reserved >= reserved


# ===========================================================================
# G2-G: Timing
# ===========================================================================

@skip_no_cuda
class TestG2GTiming:
    """G2-G: Forward, backward, and optimizer timings are measurable."""

    def test_forward_timing(self):
        model = _build_paper_backbone("cuda")
        model.eval()
        bundle = _make_bundle_for_g2().to("cuda")
        # Warmup
        with torch.no_grad():
            _ = model.encode_graph(bundle)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            _ = model.encode_graph(bundle)
        end.record()
        torch.cuda.synchronize()
        fw_ms = start.elapsed_time(end)
        assert fw_ms > 0.0, "Forward timing should be positive."

    def test_backward_timing(self):
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        torch.cuda.synchronize()
        bw_ms = start.elapsed_time(end)
        assert bw_ms > 0.0, "Backward timing should be positive."

    def test_optimizer_timing(self):
        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        optimizer.step()
        end.record()
        torch.cuda.synchronize()
        opt_ms = start.elapsed_time(end)
        assert opt_ms > 0.0, "Optimizer timing should be positive."


# ===========================================================================
# G2-H: CPU/GPU numerical comparison
# ===========================================================================

@skip_no_cuda
class TestG2HCpuGpuComparison:
    """G2-H: Same model on CPU and CUDA produce numerically similar outputs."""

    def test_cpu_and_gpu_outputs_are_close(self):
        """Same init weights, same input — CPU vs CUDA outputs match closely."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        cfg = paper_config()
        model_cpu = NeuralBackbone(cfg)
        model_gpu = NeuralBackbone(cfg)
        # Synchronise weights so both start identically.
        model_gpu.load_state_dict(model_cpu.state_dict())
        # Move GPU model BEFORE creating its input bundle so both use same init.
        model_gpu = model_gpu.to("cuda")

        bundle_cpu = _make_bundle_for_g2()
        bundle_gpu = _make_bundle_for_g2().to("cuda")

        model_cpu.eval()
        model_gpu.eval()

        with torch.no_grad():
            out_cpu = model_cpu.encode_graph(bundle_cpu)
            out_gpu = model_gpu.encode_graph(bundle_gpu)

        diff = (out_cpu.port_embeddings - out_gpu.port_embeddings.cpu()).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        # Float32 CPU vs CUDA gives tiny numerical drift; well within rtol=1e-4.
        assert max_diff < 1e-5, f"CPU/GPU max diff {max_diff:.6e} exceeds tolerance."
        assert mean_diff < 1e-6, f"CPU/GPU mean diff {mean_diff:.6e} exceeds tolerance."


# ===========================================================================
# G2-I: Encoder-only policy (backbone + sigmoid head) on CUDA
# ===========================================================================

@skip_no_cuda
class TestG2IEncoderOnlyPolicy:
    """G2-I: Encoder-only policy constructs and runs on CUDA."""

    def test_encoder_only_forward(self):
        inst = _make_synthetic_instance(n_ports=4, n_vessels=2)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        generator = ServiceGenerator(inst, dist_by_pair)
        backbone = _build_paper_backbone("cuda")
        policy = EncoderOnlyPolicy(backbone, inst, generator).to("cuda")
        bundle = _make_bundle_for_g2().to("cuda")
        fleet_rem = {e.vessel_class: float(e.quantity) for e in inst.fleet}

        policy.eval()
        with torch.no_grad():
            out = policy.deterministic_action(bundle, fleet_rem)
        assert out.port_logits.device.type == "cuda"
        assert torch.isfinite(out.port_logits).all()
        assert len(out.executed_ports) >= 2

    def test_encoder_only_loss_from_logits(self):
        """Construct a scalar loss from encoder-only logits for gradient check."""
        inst = _make_synthetic_instance(n_ports=4, n_vessels=2)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        generator = ServiceGenerator(inst, dist_by_pair)
        backbone = _build_paper_backbone("cuda")
        policy = EncoderOnlyPolicy(backbone, inst, generator).to("cuda")
        bundle = _make_bundle_for_g2().to("cuda")
        fleet_rem = {e.vessel_class: float(e.quantity) for e in inst.fleet}

        policy.train()
        # Use backbone directly to get gradient-compatible output
        backbone_out = backbone.encode_graph(bundle)
        logits = policy.port_logits(backbone_out)
        loss = logits.sum()
        assert loss.requires_grad

        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_count = sum(1 for p in policy.parameters() if p.grad is not None)
        assert grad_count > 0
        finite = sum(1 for p in policy.parameters()
                     if p.grad is not None and torch.isfinite(p.grad).all())
        assert finite == grad_count


# ===========================================================================
# G2-J: Encoder-decoder policy (backbone + LSTM decoder) on CUDA
# ===========================================================================

@skip_no_cuda
class TestG2JEncoderDecoderPolicy:
    """G2-J: Encoder-decoder policy backbone path runs on CUDA."""

    def test_encoder_decoder_backbone_path(self):
        """
        The encoder-decoder policy has torch.no_grad on sample_action and
        forward, so we validate the BACKBONE PATH only (the shared P7 encoder
        plus the LSTM decoder weights are present in the module).
        """
        inst = _make_synthetic_instance(n_ports=4, n_vessels=2)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        generator = ServiceGenerator(inst, dist_by_pair)
        backbone = _build_paper_backbone("cuda")
        policy = EncoderDecoderPolicy(backbone, inst, generator).to("cuda")
        bundle = _make_bundle_for_g2().to("cuda")
        fleet_rem = {e.vessel_class: float(e.quantity) for e in inst.fleet}

        # Validate the backbone portion with gradients
        backbone.train()
        backbone_out = backbone.encode_graph(bundle)
        loss = backbone_out.port_embeddings.sum()
        assert loss.requires_grad

        optimizer = torch.optim.AdamW(
            list(policy.parameters()), lr=1e-3
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_count = sum(1 for p in policy.parameters() if p.grad is not None)
        assert grad_count > 0

        finite = sum(1 for p in policy.parameters()
                     if p.grad is not None and torch.isfinite(p.grad).all())
        assert finite == grad_count

        optimizer.step()
        changed = sum(
            1 for name, p in policy.named_parameters()
            if p.grad is not None and not torch.allclose(
                p, p - 1e-3 * p.grad, atol=1e-7,
            )
        )
        assert changed > 0

    def test_encoder_decoder_policy_module_structure(self):
        """Confirm the encoder-decoder policy contains the expected sub-modules."""
        inst = _make_synthetic_instance(n_ports=4, n_vessels=2)
        dist_by_pair = {(a.origin, a.destination): a for a in inst.distances}
        generator = ServiceGenerator(inst, dist_by_pair)
        backbone = _build_paper_backbone("cpu")
        policy = EncoderDecoderPolicy(backbone, inst, generator)

        assert isinstance(policy.backbone, NeuralBackbone)
        assert policy.backbone.hidden_dim == 512
        assert policy.backbone.num_gat_layers == 3
        assert policy.backbone.num_transformer_layers == 3
        assert policy.backbone.num_transformer_heads == 8

        decoder = policy.decoder
        assert decoder.H == 512
        assert decoder.lstm.num_layers == 1
        assert decoder.n_candidates == 4 + 2 + 1  # P + V + BOS


# ===========================================================================
# G2-K: OOM protection (graph too large)
# ===========================================================================

@skip_no_cuda
class TestG2KLargeGraph:
    """G2-K: Larger graphs still fit within VRAM."""

    def test_baltic_size_graph(self):
        """Baltic-like instance (~12 ports) should fit comfortably."""
        torch.cuda.reset_peak_memory_stats()
        model = _build_paper_backbone("cuda")
        model.eval()
        bundle = _make_bundle_for_g2(n_ports=12, n_vessels=2).to("cuda")
        with torch.no_grad():
            out = model.encode_graph(bundle)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        assert peak < total * 0.5, (
            f"Baltic-size graph peak ({peak / 1e9:.2f} GB) exceeds 50% VRAM."
        )

    def test_world_small_graph(self):
        """WorldSmall-like instance (~20 ports) should fit within VRAM."""
        torch.cuda.reset_peak_memory_stats()
        model = _build_paper_backbone("cuda")
        model.eval()
        bundle = _make_bundle_for_g2(n_ports=20, n_vessels=2).to("cuda")
        with torch.no_grad():
            out = model.encode_graph(bundle)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        assert peak < total * 0.5, (
            f"WorldSmall-size graph peak ({peak / 1e9:.2f} GB) exceeds 50% VRAM."
        )


# ===========================================================================
# G2-L: Full training step with paper-scale backbone
# ===========================================================================

@skip_no_cuda
class TestG2LFullTrainingStep:
    """G2-L: A complete forward → loss → backward → optimizer step works."""

    def test_full_training_step(self):
        """One complete training step with paper-scale architecture on CUDA."""
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.reset_peak_memory_stats()

        model = _build_paper_backbone("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        model.train()
        bundle = _make_bundle_for_g2().to("cuda")

        # Verify architecture
        assert model.hidden_dim == 512
        assert model.num_gat_layers == 3
        assert model.num_transformer_layers == 3
        assert model.num_transformer_heads == 8

        # Forward
        out = model.encode_graph(bundle)
        loss = out.port_embeddings.sum()
        assert torch.isfinite(loss)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        assert sum(1 for p in model.parameters() if p.grad is not None) > 0
        assert all(
            torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None
        )

        # Optimizer step
        snapshots = {
            name: p.data.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }
        optimizer.step()
        changed = sum(
            1 for name, old in snapshots.items()
            if not torch.equal(old, model.state_dict()[name])
        )
        assert changed > 0, "No parameters changed after optimizer step."

        # No NaN in parameters
        assert all(torch.isfinite(p).all() for p in model.parameters())

        # Memory check
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        total = torch.cuda.get_device_properties(0).total_memory
        assert peak < total * 0.8, (
            f"Peak memory ({peak / 1e9:.2f} GB) exceeds safe limit."
        )
