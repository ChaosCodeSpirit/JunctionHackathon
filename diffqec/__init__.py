"""DiffQEC decoder package."""
from .diffusion import cosine_schedule, sample_xt, sample_x_prev
from .model import DiffQEC
from .decoder import DiffQECDecoder, load_decoder
from .integrate import decode_hardware_results_diffqec
from .data import (
    generate_dem_samples,
    generate_noisy_dem_samples,
    make_test_circuit,
    make_noisy_circuit,
    noise_sweep_table,
    ParityDataset,
)
from .noise import (
    IQM_EMERALD_TYPICAL,
    IQM_GARNET_TYPICAL,
    NoiseModel,
    apply_noise_to_circuit,
    make_emerald_noise,
    make_garnet_noise,
    noise_model_from_calibration,
)

__all__ = [
    "cosine_schedule",
    "sample_xt",
    "sample_x_prev",
    "DiffQEC",
    "DiffQECDecoder",
    "load_decoder",
    "decode_hardware_results_diffqec",
    "generate_dem_samples",
    "make_test_circuit",
    "ParityDataset",
    # Noisy extension
    "NoiseModel",
    "IQM_EMERALD_TYPICAL",
    "IQM_GARNET_TYPICAL",
    "noise_model_from_calibration",
    "apply_noise_to_circuit",
    "make_emerald_noise",
    "make_garnet_noise",
    "make_noisy_circuit",
    "generate_noisy_dem_samples",
    "noise_sweep_table",
]
