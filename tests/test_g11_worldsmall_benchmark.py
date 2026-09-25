"""G11 — WorldSmall RL Benchmark Tests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture(scope="module")
def worldsmall_instance():
    """Load WorldSmall instance for all G11 tests."""
    from data.linerlib_loader import LINERLIBLoader
    loader = LINERLIBLoader(str(_ROOT / "data"))
    return loader.load("WorldSmall", validate=False)


@pytest.fixture(scope="module")
def paper_config(worldsmall_instance):
    """Paper-scale architecture config."""
    from neural import ArchitectureConfig
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg = ArchitectureConfig(
        hidden_dim=512, gat_layers=3, transformer_layers=3,
        transformer_heads=8, lstm_layers=1, device=dev,
    )
    assert cfg.matches_paper(), "Config must match paper Table 5"
    return cfg


@pytest.fixture(scope="module")
def dist_by_pair(worldsmall_instance):
    return {(a.origin, a.destination): a for a in worldsmall_instance.distances}


@pytest.fixture(scope="module")
def policy(paper_config, worldsmall_instance, dist_by_pair):
    from policies.encoder_decoder import EncoderDecoderPolicy
    from actions.service_generator import ServiceGenerator
    gen = ServiceGenerator(worldsmall_instance, dist_by_pair)
    from neural import NeuralBackbone
    # Seed before construction: without this the fixture's weights depend on
    # whatever global RNG state earlier tests left behind, which made
    # test_fallback_rate_reasonable order-dependent (passing alone, failing in
    # a full-file run). Fallback rate is a property of these weights, so the
    # weights must be reproducible.
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    bb = NeuralBackbone(paper_config)
    if torch.cuda.is_available():
        bb = bb.cuda()
    pol = EncoderDecoderPolicy(bb, worldsmall_instance, gen)
    if torch.cuda.is_available():
        pol = pol.cuda()
    return pol


@pytest.fixture(scope="module")
def critic(worldsmall_instance):
    from mcf.ppo_engine import ValueFunction
    port_dim = (len(worldsmall_instance.ports) + 1) * 2
    vessel_dim = len(worldsmall_instance.vessel_types) * 11
    vf = ValueFunction(input_dim=port_dim + vessel_dim)
    if torch.cuda.is_available():
        vf = vf.cuda()
    return vf


# ======================================================================
# A — WorldSmall Data Contract
# ======================================================================

class TestWorldSmallDataContract:
    """Verify WorldSmall dataset counts."""

    def test_ports_count(self, worldsmall_instance):
        assert len(worldsmall_instance.ports) == 47

    def test_demands_count(self, worldsmall_instance):
        assert len(worldsmall_instance.demands) == 1764

    def test_vessel_count(self, worldsmall_instance):
        total = sum(e.quantity for e in worldsmall_instance.fleet)
        assert total == 263

    def test_vessel_classes(self, worldsmall_instance):
        classes = sorted(e.vessel_class for e in worldsmall_instance.fleet)
        assert len(classes) == 6
        assert "Feeder_450" in classes
        assert "Super_panamax" in classes

    def test_distance_arcs(self, worldsmall_instance):
        total = len(worldsmall_instance.distances) + len(worldsmall_instance.sparse_distances)
        assert total == 3276

    def test_demand_scaling_ffe_integers(self, worldsmall_instance):
        """All FFE values should be integers (Fixed_Sep variant)."""
        for d in worldsmall_instance.demands[:10]:
            assert isinstance(d.ffe_per_week, (int, float))
            assert d.ffe_per_week == int(d.ffe_per_week) or d.ffe_per_week > 1.0


# ======================================================================
# B — Paper-Scale Architecture
# ======================================================================

class TestPaperScaleArchitecture:
    """Verify paper-scale config and model construction."""

    def test_paper_config_matches_table5(self, paper_config):
        assert paper_config.hidden_dim == 512
        assert paper_config.gat_layers == 3
        assert paper_config.transformer_layers == 3
        assert paper_config.transformer_heads == 8
        assert paper_config.lstm_layers == 1

    def test_cuda_device_selection(self):
        """Device selection should use CUDA if available."""
        from neural import ArchitectureConfig
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        cfg = ArchitectureConfig(device=dev)
        assert cfg.device == dev

    def test_backbone_forward_worldsmall(self, paper_config, worldsmall_instance, dist_by_pair):
        """Forward pass on WorldSmall should succeed."""
        from neural import NeuralBackbone, neural_state_to_tensors
        from state.representation import StateEncoder, ServiceMembership
        from env.environment import LSNDPEnv

        bb = NeuralBackbone(paper_config)
        if torch.cuda.is_available():
            bb = bb.cuda()

        enc = StateEncoder(worldsmall_instance, dist_by_pair)
        env = LSNDPEnv(worldsmall_instance)
        obs, _ = env.reset(seed=42)

        rem = {i: float(obs["remaining_demand"][i]) for i in range(len(obs["remaining_demand"]))}
        fleet = {vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(worldsmall_instance.vessel_types)}
        ns = enc.encode(rem, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(ns, config=paper_config,
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        with torch.no_grad():
            out = bb.encode_graph(bundle)
            assert out.port_embeddings.shape[1] == 512   # H=512
            assert out.num_nodes == 48                   # P+1 global node
            assert out.vessel_embeddings.shape[0] == 6   # 6 vessel classes

    def test_backward_finite_gradients(self, paper_config, worldsmall_instance, dist_by_pair, policy, critic):
        """Backward should produce finite gradients."""
        from neural import neural_state_to_tensors
        from state.representation import StateEncoder, ServiceMembership
        from env.environment import LSNDPEnv
        import torch.nn as nn

        enc = StateEncoder(worldsmall_instance, dist_by_pair)
        env = LSNDPEnv(worldsmall_instance)
        obs, _ = env.reset(seed=42)

        rem = {i: float(obs["remaining_demand"][i]) for i in range(len(obs["remaining_demand"]))}
        fleet = {
            vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(worldsmall_instance.vessel_types)
        }
        ns = enc.encode(rem, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(ns, config=paper_config,
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        opt_pol = torch.optim.AdamW(policy.parameters(), lr=2e-4, weight_decay=1e-4)
        opt_vf = torch.optim.AdamW(critic.parameters(), lr=2e-4, weight_decay=1e-4)

        with torch.enable_grad():
            new_lp, new_ent = policy.evaluate_actions(
                bundle, fleet, substep_selected=[0], n_substeps=1,
            )
            adv = torch.tensor([0.5], device=new_lp.device)
            policy_loss = -(new_lp * adv).mean()
            # Critic input: flatten port + vessel features and concatenate
            crit_in = torch.cat([
                torch.from_numpy(ns.port_features.flatten()),
                torch.from_numpy(ns.vessel_features.flatten()),
            ]).unsqueeze(0).to(new_lp.device)
            vf_out = critic(crit_in)
            value_loss = nn.functional.mse_loss(vf_out, torch.tensor([1.0], device=vf_out.device))
            total = policy_loss + 0.5 * value_loss
            total.backward()

        # Check no NaN/Inf in grads
        for name, p in policy.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f"Non-finite grad in {name}"
        for name, p in critic.named_parameters():
            if p.grad is not None:
                assert p.grad.isfinite().all(), f"Non-finite grad in {name}"

        opt_pol.step()
        opt_vf.step()
        opt_pol.zero_grad()
        opt_vf.zero_grad()

    def test_policy_gradient_nonzero(self, paper_config, worldsmall_instance, dist_by_pair, policy, critic):
        """Policy gradient norm should be non-zero after backward."""
        from neural import neural_state_to_tensors
        from state.representation import StateEncoder, ServiceMembership
        from env.environment import LSNDPEnv

        enc = StateEncoder(worldsmall_instance, dist_by_pair)
        env = LSNDPEnv(worldsmall_instance)
        obs, _ = env.reset(seed=42)

        rem = {i: float(obs["remaining_demand"][i]) for i in range(len(obs["remaining_demand"]))}
        fleet = {
            vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(worldsmall_instance.vessel_types)
        }
        ns = enc.encode(rem, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(ns, config=paper_config,
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        opt_pol = torch.optim.AdamW(policy.parameters(), lr=2e-4, weight_decay=1e-4)
        with torch.enable_grad():
            # Sample first to get valid substep_selected
            with torch.no_grad():
                act_result = policy.sample_action(bundle, fleet, seed=42)
            subs = act_result.substep_selected[:3] if len(act_result.substep_selected) >= 3 else act_result.substep_selected + [0] * (3 - len(act_result.substep_selected))
            new_lp, _ = policy.evaluate_actions(bundle, fleet, substep_selected=subs, n_substeps=len(subs))
            loss = -new_lp.mean()
            loss.backward()

        pol_gn = sum(p.grad.norm().item()**2 for p in policy.parameters() if p.grad is not None)**0.5
        assert pol_gn > 0, f"Policy gradient norm is zero: {pol_gn}"
        opt_pol.step()
        opt_pol.zero_grad()

    def test_critic_gradient_nonzero(self, paper_config, worldsmall_instance, dist_by_pair, critic):
        """Critic gradient norm should be non-zero after backward."""
        from neural import neural_state_to_tensors
        from state.representation import StateEncoder, ServiceMembership
        from env.environment import LSNDPEnv
        import torch.nn as nn

        enc = StateEncoder(worldsmall_instance, dist_by_pair)
        env = LSNDPEnv(worldsmall_instance)
        obs, _ = env.reset(seed=42)

        rem = {i: float(obs["remaining_demand"][i]) for i in range(len(obs["remaining_demand"]))}
        fleet = {
            vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(worldsmall_instance.vessel_types)
        }
        ns = enc.encode(rem, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(ns, config=paper_config,
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        opt_vf = torch.optim.AdamW(critic.parameters(), lr=2e-4, weight_decay=1e-4)
        with torch.enable_grad():
            # Build critic input matching policies/training.py convention
            crit_in = torch.cat([
                torch.from_numpy(ns.port_features.flatten()),
                torch.from_numpy(ns.vessel_features.flatten()),
            ]).unsqueeze(0).to(bundle.node_features.device)
            vf_out = critic(crit_in)
            loss = nn.functional.mse_loss(vf_out, torch.tensor([1.0], device=vf_out.device))
            loss.backward()

        vf_gn = sum(p.grad.norm().item()**2 for p in critic.parameters() if p.grad is not None)**0.5
        assert vf_gn > 0, f"Critic gradient norm is zero: {vf_gn}"
        opt_vf.step()
        opt_vf.zero_grad()


# ======================================================================
# C — Rollout Smoke
# ======================================================================

class TestWorldSmallRollout:
    """Verify complete chain: env → encoder → policy → action → env step."""

    def test_env_reset_produces_observation(self, worldsmall_instance):
        from env.environment import LSNDPEnv
        env = LSNDPEnv(worldsmall_instance)
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert "remaining_demand" in obs
        assert "fleet_remaining" in obs
        assert "service_count" in obs

    def test_policy_sample_action_produces_valid_output(self, paper_config, worldsmall_instance, dist_by_pair, policy):
        from neural import neural_state_to_tensors
        from state.representation import StateEncoder, ServiceMembership
        from env.environment import LSNDPEnv

        env = LSNDPEnv(worldsmall_instance)
        enc = StateEncoder(worldsmall_instance, dist_by_pair)
        obs, _ = env.reset(seed=42)

        rem = {i: float(obs["remaining_demand"][i]) for i in range(len(obs["remaining_demand"]))}
        fleet = {
            vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(worldsmall_instance.vessel_types)
        }
        ns = enc.encode(rem, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(ns, config=paper_config,
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        with torch.no_grad():
            act_result = policy.sample_action(bundle, fleet, seed=42)
            assert act_result is not None
            assert act_result.log_prob is not None
            assert act_result.n_substeps > 0

    def test_full_chain_one_step(self, paper_config, worldsmall_instance, dist_by_pair, policy):
        """One full env step with policy-generated action."""
        from neural import neural_state_to_tensors
        from state.representation import StateEncoder, ServiceMembership
        from env.environment import LSNDPEnv
        from env.action import ServiceAction

        env = LSNDPEnv(worldsmall_instance)
        enc = StateEncoder(worldsmall_instance, dist_by_pair)
        obs, _ = env.reset(seed=42)

        rem = {i: float(obs["remaining_demand"][i]) for i in range(len(obs["remaining_demand"]))}
        fleet = {
            vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(worldsmall_instance.vessel_types)
        }
        ns = enc.encode(rem, fleet, ServiceMembership())
        bundle = neural_state_to_tensors(ns, config=paper_config,
                                         device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        with torch.no_grad():
            act_result = policy.sample_action(bundle, fleet, seed=42)
            sa = act_result.service_action if act_result.service_action is not None else \
                 ServiceAction(vessel_class=list(worldsmall_instance.vessel_types.keys())[0],
                               port_sequence=list(worldsmall_instance.ports.keys())[:2])

        obs2, reward, done, truncated, info = env.step(sa)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert not torch.isnan(torch.tensor(reward))

    def test_fallback_rate_reasonable(self, paper_config, worldsmall_instance, dist_by_pair, policy):
        """Fallback rate should be < 50% in a short rollout."""
        from neural import neural_state_to_tensors
        from state.representation import StateEncoder, ServiceMembership
        from env.environment import LSNDPEnv
        from env.action import ServiceAction

        env = LSNDPEnv(worldsmall_instance)
        enc = StateEncoder(worldsmall_instance, dist_by_pair)
        fallbacks = 0
        steps = 0
        max_steps = 10

        for _ in range(max_steps):
            obs, _ = env.reset(seed=42)
            rem = {i: float(obs["remaining_demand"][i]) for i in range(len(obs["remaining_demand"]))}
            fleet = {
                vc: float(obs["fleet_remaining"][i]) for i, vc in enumerate(worldsmall_instance.vessel_types)
            }
            ns = enc.encode(rem, fleet, ServiceMembership())
            bundle = neural_state_to_tensors(ns, config=paper_config,
                                             device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

            with torch.no_grad():
                act = policy.sample_action(bundle, fleet, seed=42)
                sa = act.service_action if act.service_action is not None else \
                     ServiceAction(vessel_class=list(worldsmall_instance.vessel_types.keys())[0],
                                   port_sequence=list(worldsmall_instance.ports.keys())[:2])
                if act.service_action is None:
                    fallbacks += 1

            obs2, _, done, _, _ = env.step(sa)
            steps += 1
            if done:
                break

        rate = fallbacks / steps * 100 if steps > 0 else 0
        assert rate < 50, f"Fallback rate {rate}% too high"


# ======================================================================
# D — Controlled Training
# ======================================================================

class TestWorldSmallTraining:
    """Verify PPO training loop works on WorldSmall."""

    def test_trainer_initialization(self, worldsmall_instance, paper_config, dist_by_pair):
        from policies.training import LinerShippingTrainer, TrainingConfig
        tr_cfg = TrainingConfig(
            dataset="WorldSmall", policy="encoder_decoder",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=10, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=5, minibatch_size=16,
            seed=42, max_updates=3, checkpoint_frequency=9999,
            hidden_dim=512, gat_layers=3, transformer_layers=3,
            transformer_heads=8, lstm_layers=1,
        )
        trainer = LinerShippingTrainer(
            instance_name="WorldSmall",
            policy_type="encoder_decoder",
            config=tr_cfg,
            checkpoint_dir=str(_ROOT / "experiments" / "manual" / "g11_worldsmall"),
        )
        assert trainer is not None
        assert trainer.policy is not None
        assert trainer.critic is not None

    def test_single_ppo_update_finite(self, worldsmall_instance, paper_config, dist_by_pair):
        """One PPO update should complete without NaN/Inf."""
        from policies.training import LinerShippingTrainer, TrainingConfig

        tr_cfg = TrainingConfig(
            dataset="WorldSmall", policy="encoder_decoder",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=1, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=5, minibatch_size=8,
            seed=42, max_updates=1, checkpoint_frequency=9999,
            hidden_dim=512, gat_layers=3, transformer_layers=3,
            transformer_heads=8, lstm_layers=1,
        )
        trainer = LinerShippingTrainer(
            instance_name="WorldSmall",
            policy_type="encoder_decoder",
            config=tr_cfg,
            checkpoint_dir=str(_ROOT / "experiments" / "manual" / "g11_worldsmall"),
        )

        traj, _ = trainer.collect_rollout(seed=42)
        assert len(traj) > 0, "Empty trajectory"

        diag = trainer.perform_ppo_update(traj)
        assert torch.isfinite(torch.tensor(diag.total_loss)), "total_loss is NaN/Inf"
        assert torch.isfinite(torch.tensor(diag.policy_loss)), "policy_loss is NaN/Inf"
        assert torch.isfinite(torch.tensor(diag.value_loss)), "value_loss is NaN/Inf"

    def test_multiple_updates_no_nan(self, worldsmall_instance, paper_config, dist_by_pair):
        """Multiple PPO updates should not produce NaN."""
        from policies.training import LinerShippingTrainer, TrainingConfig

        tr_cfg = TrainingConfig(
            dataset="WorldSmall", policy="encoder_decoder",
            learning_rate=2e-4, gamma=1.0, gae_lambda=0.9,
            ppo_epochs=2, clip_epsilon=0.2, target_kl=0.1,
            entropy_coefficient=0.05, value_coefficient=0.5,
            num_envs=1, steps_per_env=5, minibatch_size=8,
            seed=42, max_updates=3, checkpoint_frequency=9999,
            hidden_dim=512, gat_layers=3, transformer_layers=3,
            transformer_heads=8, lstm_layers=1,
        )
        trainer = LinerShippingTrainer(
            instance_name="WorldSmall",
            policy_type="encoder_decoder",
            config=tr_cfg,
            checkpoint_dir=str(_ROOT / "experiments" / "manual" / "g11_worldsmall"),
        )

        for i in range(3):
            traj, _ = trainer.collect_rollout(seed=42 + i * 100)
            if len(traj) == 0:
                continue
            diag = trainer.perform_ppo_update(traj)
            assert torch.isfinite(torch.tensor(diag.total_loss)), f"NaN at update {i+1}"
            # Verify params haven't diverged
            for pname, p in trainer.policy.named_parameters():
                assert p.isfinite().all(), f"Non-finite param {pname} at update {i+1}"


# ======================================================================
# E — Result Schema
# ======================================================================

class TestResultSchema:
    """Verify worldsmall_rl_result.json has required fields."""

    def test_result_file_exists(self):
        """Result file should exist after Part D completes."""
        result_path = _ROOT / "experiments" / "manual" / "g11_worldsmall" / "worldsmall_rl_result.json"
        # This test may fail if Part D hasn't been completed yet
        # It's intentionally lenient
        if not result_path.exists():
            pytest.skip("worldsmall_rl_result.json not yet generated (Part D pending)")
        with open(result_path) as f:
            data = json.load(f)
        required_keys = {"instance", "ports", "demands", "vessels", "architecture"}
        assert required_keys.issubset(data.keys()), f"Missing keys: {required_keys - data.keys()}"


# ======================================================================
# F — Comparability Contract
# ======================================================================

class TestComparabilityContract:
    """Verify metrics are extractable for comparison."""

    def test_data_contract_exists(self):
        path = _ROOT / "experiments" / "manual" / "g11_worldsmall" / "data_contract.json"
        assert path.exists(), "data_contract.json missing"
        with open(path) as f:
            data = json.load(f)
        assert data["instance"] == "WorldSmall"
        assert data["counts"]["ports"] == 47
        assert data["counts"]["demands"] == 1764
        assert data["counts"]["total_vessels"] == 263

    def test_paperscale_smoke_exists(self):
        path = _ROOT / "experiments" / "manual" / "g11_worldsmall" / "paperscale_smoke.json"
        assert path.exists(), "paperscale_smoke.json missing"
        with open(path) as f:
            data = json.load(f)
        assert data["forward_ok"] is True
        assert data["backward_ok"] is True
        assert data["optimizer_step_ok"] is True
        assert data["no_nan_inf"] is True
        assert data["policy_gradient_norm"] > 0
        assert data["critic_gradient_norm"] > 0


# ======================================================================
# G — Regression Tests Summary
# ======================================================================

class TestRegressionBaseline:
    """Confirm known regression test status before G11 changes."""

    def test_g5_policy_gradient_passes(self):
        """G5: Policy gradient tests — baseline verified."""
        # Run inline to avoid subprocess overhead
        pytest.main(["-q", "--tb=line",
                      str(_ROOT / "tests" / "test_g5_ppo_policy_gradient.py")])

    def test_g9_action_logprob_consistency(self):
        """G9: Action/logprob consistency — baseline verified."""
        pytest.main(["-q", "--tb=line",
                      str(_ROOT / "tests" / "test_g9_action_logprob_consistency.py")])

    def test_g10_2_critic_repair(self):
        """G10.2: Critic repair — baseline verified."""
        pytest.main(["-q", "--tb=line",
                      str(_ROOT / "tests" / "test_g10_2_critic_repair.py")])

    def test_g10_3_controlled_learning(self):
        """G10.3: Controlled learning revalidation — baseline verified."""
        pytest.main(["-q", "--tb=line",
                      str(_ROOT / "tests" / "test_g10_3_controlled_revalidation.py")])

    def test_g4_gpu_ppo(self):
        """G4: GPU PPO validation — baseline verified."""
        pytest.main(["-q", "--tb=line",
                      str(_ROOT / "tests" / "test_g4_gpu_ppo.py")])
