"""Tests for the noisy extension of the DiffQEC pipeline."""
import json
from pathlib import Path

import numpy as np
import pytest
import stim

from diffqec.data import (
    ParityDataset,
    generate_noisy_dem_samples,
    make_noisy_circuit,
    noise_sweep_table,
    reshape_syndrome,
)
from diffqec.noise import (
    IQM_EMERALD_TYPICAL,
    IQM_GARNET_TYPICAL,
    NoiseModel,
    apply_noise_to_circuit,
    make_emerald_noise,
    make_garnet_noise,
    noise_model_from_calibration,
)

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch not installed in this environment",
)


class TestNoiseModel:
    def test_default_construction(self):
        n = NoiseModel()
        assert 0.0 <= n.p_1q <= 1.0
        assert 0.0 <= n.p_2q <= 1.0
        assert 0.0 <= n.p_meas <= 1.0
        assert 0.0 <= n.p_reset <= 1.0
        assert 0.0 <= n.p_idle <= 1.0

    def test_scale(self):
        n = NoiseModel(p_1q=0.01, p_2q=0.05, p_meas=0.02, p_reset=0.01, p_idle=0.005)
        n2 = n.scale(0.5)
        assert n2.p_1q == pytest.approx(0.005)
        assert n2.p_2q == pytest.approx(0.025)
        assert n2.p_meas == pytest.approx(0.01)
        assert n2.p_reset == pytest.approx(0.005)
        assert n2.p_idle == pytest.approx(0.0025)
        assert n.scale(0.0).p_1q == 0.0

    def test_scale_rejects_negative(self):
        with pytest.raises(ValueError):
            NoiseModel().scale(-0.1)

    def test_iqm_typical_models_are_sane(self):
        for m in (IQM_EMERALD_TYPICAL, IQM_GARNET_TYPICAL):
            assert m.p_2q > m.p_1q  # 2q is the dominant error
            assert m.p_meas > 0
            assert m.p_idle > 0
            assert m.name.startswith("iqm_")

    def test_calibration_fallback(self):
        m = noise_model_from_calibration({})
        # Falls back to the IQM Emerald typical values
        assert m.p_1q == IQM_EMERALD_TYPICAL.p_1q
        assert m.p_2q == IQM_EMERALD_TYPICAL.p_2q

    def test_calibration_ingests_averaging(self):
        cal = {
            "single_qubit_gate_error": [0.001, 0.002, 0.003],
            "two_qubit_gate_error":    [0.01, 0.02],
            "readout_error":           [0.005, 0.015],
            "reset_error":             0.004,
            "t1_us":                   30.0,
            "t2_us":                   20.0,
            "measure_time_ns":         1000.0,
        }
        m = noise_model_from_calibration(cal)
        assert m.p_1q == pytest.approx(0.002)
        assert m.p_2q == pytest.approx(0.015)
        assert m.p_meas == pytest.approx(0.01)
        assert m.p_reset == pytest.approx(0.004)
        # Idle derived from T1/T2 + readout time
        assert 0.0 < m.p_idle < 0.5

    def test_convenience_factories(self):
        a = make_emerald_noise()
        b = make_emerald_noise(scale=0.0)
        assert a == IQM_EMERALD_TYPICAL
        assert b.p_1q == 0.0 and b.p_2q == 0.0
        g = make_garnet_noise(scale=2.0)
        assert g.p_2q == pytest.approx(2.0 * IQM_GARNET_TYPICAL.p_2q)


class TestNoisyCircuit:
    def test_noiseless_when_noise_is_none(self):
        c = make_noisy_circuit(distance=3, rounds=3, noise=None)
        # No DEPOLARIZE / X_ERROR / Z_ERROR should be present.
        flat = list(c.flattened())
        for inst in flat:
            assert inst.name not in (
                "DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR",
            ), f"unexpected noise op {inst.name} in noiseless circuit"

    def test_noisy_circuit_injects_error_channels(self):
        n = NoiseModel(p_1q=0.01, p_2q=0.02, p_meas=0.01, p_reset=0.01, p_idle=0.01)
        c = make_noisy_circuit(distance=3, rounds=3, noise=n)
        flat_names = {i.name for i in c.flattened()}
        assert "DEPOLARIZE1" in flat_names
        assert "DEPOLARIZE2" in flat_names
        assert "X_ERROR" in flat_names
        assert "Z_ERROR" in flat_names

    def test_noisy_circuit_z_error_count_matches_rounds(self):
        # For d=3, rounds=r, the noisy circuit should inject one Z_ERROR per
        # round, i.e. (r+1) in total: the initial round, the (r-1) repeated
        # rounds, and the final data readout.
        n = IQM_EMERALD_TYPICAL
        for r in (2, 4, 7):
            c = make_noisy_circuit(distance=3, rounds=r, noise=n)
            n_zerr = sum(1 for i in c.flattened() if i.name == "Z_ERROR")
            assert n_zerr == r + 1, f"rounds={r}: expected {r+1} Z_ERRORs, got {n_zerr}"

    def test_apply_noise_to_circuit_does_not_mutate_input(self):
        base = make_noisy_circuit(distance=3, rounds=3, noise=None)
        n = NoiseModel(p_2q=0.1)
        noisy = apply_noise_to_circuit(base, n)
        assert not any(i.name.startswith("DEPOLARIZE") for i in base.flattened())
        assert any(i.name == "DEPOLARIZE2" for i in noisy.flattened())

    def test_dem_samples_have_expected_shape(self):
        c, det, obs = generate_noisy_dem_samples(
            distance=3, rounds=3, shots=50, seed=7,
        )
        assert isinstance(c, stim.Circuit)
        assert det.shape == (50, c.num_detectors)
        assert obs.shape == (50, c.num_observables)
        assert det.dtype == bool
        assert obs.dtype == bool


class TestNoiseSweep:
    def test_noise_sweep_table_shape(self):
        rows = noise_sweep_table(
            distance=3, rounds=3, shots=80,
            scales=(0.5, 1.0, 2.0), seed=11,
        )
        assert len(rows) == 3
        assert [r["scale"] for r in rows] == [0.5, 1.0, 2.0]
        for r in rows:
            assert r["det_events"].shape[0] == 80
            assert 0.0 <= r["mean_detector_firing_rate"] <= 1.0

    def test_higher_scale_increases_firing_rate(self):
        rows = noise_sweep_table(
            distance=3, rounds=3, shots=400,
            scales=(0.0, 2.0), seed=23,
        )
        assert rows[0]["mean_detector_firing_rate"] < rows[1]["mean_detector_firing_rate"]


class TestNoisyTraining:
    @requires_torch
    def test_train_and_decode_noisy(self):
        import torch
        from diffqec.model import DiffQEC
        from diffqec.training import train_diffqec
        from diffqec.decoder import DiffQECDecoder
        c, det, obs = generate_noisy_dem_samples(
            distance=3, rounds=3, shots=80, seed=42,
        )
        syndrome = reshape_syndrome(det, c, target_rounds=3)
        L = obs.shape[1]
        model = DiffQEC(L=L, syndrome_shape=syndrome.shape[1:], hidden=16, num_denoiser_layers=1)
        train_ds = ParityDataset(det, obs, parity="even", target_rounds=3)
        loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True)
        model = train_diffqec(
            model, loader, T=10, epochs=2, lr=1e-3,
            device=torch.device("cpu"), checkpoint_dir=None,
        )
        decoder = DiffQECDecoder(model, num_steps=10)
        pred, conf = decoder.decode_syndromes({"det_events": det, "obs_flips": obs})
        assert pred.shape == (80, L)
        assert np.isfinite(conf).all()


class TestTrainingScript:
    @requires_torch
    def test_sweep_output_is_valid_json(self, tmp_path: Path):
        import torch
        from scripts.train_noisy_diffqec import _train_one
        m = _train_one(
            distance=3, rounds=3, shots=80,
            noise=IQM_EMERALD_TYPICAL, scale=1.0,
            memory="Z", no_reset=False, seed=0,
            device=torch.device("cpu"),
            epochs=1, lr=1e-3, T=10, hidden=16, num_denoiser_layers=1,
            batch_size=16, checkpoint_dir=None,
        )
        s = json.dumps(m, indent=2)
        d = json.loads(s)
        assert "ler" in d and "stderr" in d
        assert d["scale"] == 1.0
        assert d["distance"] == 3
