"""Tests for DiffQEC decoder."""
import math
import numpy as np
import pytest
import torch

# Module imports
try:
    from diffqec.diffusion import cosine_schedule, sample_xt, reverse_posterior_prob, sample_x_prev, get_x0_from_logits
    from diffqec.model import DiffQEC
    from diffqec.syndrome_processor import SyndromeProcessor
    from diffqec.denoiser import SFMLayer, DenoiserStack
    from diffqec.data import make_test_circuit, generate_dem_samples, reshape_syndrome, ParityDataset
    from diffqec.decoder import DiffQECDecoder
    from diffqec.integrate import decode_hardware_results_diffqec
except ImportError as e:
    import sys
    print(f"Import error: {e}")
    sys.exit(1)


class TestDiffusionMath:
    def test_cosine_schedule_bounds(self):
        T = 50
        ab, beta, alpha = cosine_schedule(T)
        assert ab[0] == 1.0
        assert ab[-1] < 0.01
        assert np.all(beta >= 1e-4)
        assert np.all(beta <= 0.9999)
        assert np.allclose(alpha, 1.0 - beta, atol=1e-5)

    def test_forward_kernel_shape(self):
        x0 = torch.tensor([[0, 1, 0], [1, 1, 0]], dtype=torch.float32)
        t = 10
        ab, _, _ = cosine_schedule(50)
        xt = sample_xt(x0, t, ab)
        assert xt.shape == x0.shape
        assert torch.all((xt == 0.0) | (xt == 1.0))

    def test_posterior_is_valid_probability(self):
        ab, beta, _ = cosine_schedule(20)
        xt = torch.tensor([0.0, 1.0])
        x0 = torch.tensor([1.0, 0.0])
        prob = reverse_posterior_prob(xt, x0, float(beta[5]), float(ab[5]), float(ab[4]))
        assert torch.all(prob >= 0.0)
        assert torch.all(prob <= 1.0)

    def test_reverse_sample_shape(self):
        ab, beta, _ = cosine_schedule(20)
        xt = torch.rand(4, 8)
        x0_prob = torch.rand(4, 8)
        x_prev = sample_x_prev(xt, x0_prob, float(beta[5]), float(ab[5]), float(ab[4]))
        assert x_prev.shape == (4, 8)

    def test_logits_to_x0(self):
        logits = torch.tensor([[[5.0, -1.0], [-2.0, 3.0]]])  # (1, 2, 2)
        x0 = get_x0_from_logits(logits)
        assert x0.shape == (1, 2)
        assert x0[0, 0].item() == 0.0
        assert x0[0, 1].item() == 1.0


class TestModelShapes:
    def test_syndrome_processor_output(self):
        proc = SyndromeProcessor(in_spatial=16, hidden=32, out_feature=32)
        s = torch.rand(2, 5, 16)  # (batch, rounds, D)
        c = proc(s)
        assert c.shape == (2, 32)

    def test_sfm_layer(self):
        sfm = SFMLayer(feature_dim=16, cond_dim=8)
        x = torch.rand(3, 16)
        c = torch.rand(3, 8)
        y = sfm(x, c)
        assert y.shape == (3, 16)

    def test_denoiser_stack(self):
        stack = DenoiserStack(feature_dim=16, cond_dim=8, num_layers=3)
        x = torch.rand(3, 16)
        c = torch.rand(3, 8)
        y = stack(x, c)
        assert y.shape == (3, 16)

    def test_diffqec_forward(self):
        model = DiffQEC(L=1, syndrome_shape=(3, 9), hidden=16, num_denoiser_layers=2)
        xt = torch.tensor([[0.0], [1.0]])
        t = torch.tensor([5, 10])
        syndrome = torch.rand(2, 3, 9)
        logits = model(xt, t, syndrome)
        assert logits.shape == (2, 1, 2)


class TestDataPipeline:
    def test_make_test_circuit(self):
        circuit = make_test_circuit(distance=3, rounds=3, noise=0.01)
        assert circuit.num_detectors > 0
        assert circuit.num_observables > 0

    def test_generate_samples(self):
        circuit = make_test_circuit(distance=3, rounds=3, noise=0.01)
        det, obs = generate_dem_samples(circuit, shots=50, seed=123)
        assert det.shape[0] == 50
        assert obs.shape[0] == 50
        assert det.dtype == bool

    def test_reshape_syndrome(self):
        circuit = make_test_circuit(distance=3, rounds=3, noise=0.01)
        det, _ = generate_dem_samples(circuit, shots=10)
        reshaped = reshape_syndrome(det, circuit, target_rounds=3)
        assert reshaped.shape[0] == 10
        assert reshaped.shape[1] == 3

    def test_parity_dataset_split(self):
        circuit = make_test_circuit(distance=3, rounds=3, noise=0.01)
        det, obs = generate_dem_samples(circuit, shots=100)
        ds_even = ParityDataset(det, obs, parity="even")
        ds_odd = ParityDataset(det, obs, parity="odd")
        assert len(ds_even) + len(ds_odd) == 100
        assert len(ds_even) > 0
        assert len(ds_odd) > 0


class TestIntegration:
    def test_tiny_training_and_inference(self):
        """Train a tiny model for a few steps and verify decode contract."""
        circuit = make_test_circuit(distance=3, rounds=3, noise=0.05)
        det, obs = generate_dem_samples(circuit, shots=80, seed=42)
        syndrome = reshape_syndrome(det, circuit, target_rounds=3)

        L = obs.shape[1]
        model = DiffQEC(L=L, syndrome_shape=syndrome.shape[1:], hidden=16, num_denoiser_layers=1)
        train_ds = ParityDataset(det, obs, parity="even", target_rounds=3)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)

        device = torch.device("cpu")
        from diffqec.training import train_diffqec
        model = train_diffqec(model, loader, T=10, epochs=2, lr=1e-3, device=device, checkpoint_dir=None)

        # Inference via decoder API
        decoder = DiffQECDecoder(model, num_steps=10, device=device)
        syndromes_dict = {
            "det_events": det,
            "obs_flips": obs,
        }
        pred, conf = decoder.decode_syndromes(syndromes_dict)
        assert pred.shape == (80, L)
        assert conf.shape == (80,)
        assert np.isfinite(conf).all()

    def test_integration_contract(self, tmp_path):
        """Verify decode_hardware_results_diffqec returns finite (ler, err)."""
        circuit = make_test_circuit(distance=3, rounds=3, noise=0.05)
        det, obs = generate_dem_samples(circuit, shots=40, seed=7)
        syndrome = reshape_syndrome(det, circuit, target_rounds=3)
        L = obs.shape[1]
        model = DiffQEC(L=L, syndrome_shape=syndrome.shape[1:], hidden=16, num_denoiser_layers=1)
        train_ds = ParityDataset(det, obs, parity="even", target_rounds=3)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)

        from diffqec.training import train_diffqec
        model = train_diffqec(model, loader, T=10, epochs=2, lr=1e-3, device=torch.device("cpu"), checkpoint_dir=tmp_path)

        # Save checkpoint
        ckpt_path = tmp_path / "final.pt"
        torch.save({"model": model.state_dict()}, ckpt_path)

        syndromes_dict = {
            "det_events": det,
            "obs_flips": obs,
            "num_detectors": det.shape[1],
            "num_shots": len(det),
            "syndrome_weight_per_shot": det.sum(axis=1).astype(int),
            "detector_firing_rate": det.mean(axis=0),
        }
        ler, err = decode_hardware_results_diffqec(
            syndromes_dict,
            model_path=str(ckpt_path),
            syndrome_shape=syndrome.shape[1:],
            L=L,
            hidden=16,
            num_denoiser_layers=1,
        )
        assert len(ler) == L
        assert len(err) == L
        assert all(np.isfinite(v) for v in ler)
        assert all(np.isfinite(v) for v in err)
        assert all(v >= 0 for v in ler)
        assert all(v >= 0 for v in err)
