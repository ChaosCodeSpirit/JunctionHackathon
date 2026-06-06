# DiffQEC Decoder

Discrete denoising-diffusion decoder for quantum error correction,
implementing the DiffQEC algorithm (Xu et al. 2026, arXiv:2604.24640).

## Quick start

```bash
# Install dependencies (torch is added automatically by the root pyproject.toml)
pip install -e ..

# Smoke test — trains a tiny model on synthetic d=3 data (~60 s on CPU)
python -m diffqec.smoke_test

# Run unit + integration tests
pytest tests/test_diffqec.py -v
```

## Training a model

```python
from diffqec.data import make_test_circuit, generate_dem_samples, reshape_syndrome, ParityDataset
from diffqec.model import DiffQEC
from diffqec.training import train_diffqec
import torch

circuit = make_test_circuit(distance=5, rounds=5, noise=0.01)
det, obs = generate_dem_samples(circuit, shots=10_000, seed=42)
syndrome = reshape_syndrome(det, circuit, target_rounds=5)

model = DiffQEC(L=obs.shape[1], syndrome_shape=syndrome.shape[1:], hidden=64, num_denoiser_layers=4)
train_ds = ParityDataset(det, obs, parity="even")
loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

model = train_diffqec(model, loader, T=100, epochs=20, lr=3e-4, checkpoint_dir=Path("ckpt"))
```

## Decoding via the pipeline

```python
from diffqec.integrate import decode_hardware_results_diffqec

ler, err = decode_hardware_results_diffqec(
    syndromes,
    model_path="ckpt/final.pt",
    syndrome_shape=(5, 24),
    L=1,
)
```

## Selecting the decoder in `run_on_hardware.py`

```python
ler, err = decode_hardware_results(syndromes, decoder="diffqec", model_path="ckpt/final.pt")
```

## Algorithm summary

- **Forward process**: binary symmetric corruption chain with cosine schedule.
- **Reverse process**: learned denoiser conditioned on syndrome history.
- **Architecture**: spatial CNN → temporal GRU → SFM denoiser stack.
- **Training**: bitwise cross-entropy with curriculum over round counts.
- **Inference**: start from random noise, iteratively denoise to `x_0 = argmax(zθ)`.

## Files

- `diffusion.py` — discrete diffusion kernels and schedule.
- `syndrome_processor.py` — spatial + temporal syndrome encoder.
- `denoiser.py` — SFM (FiLM-style) layers with gated residuals.
- `model.py` — top-level `DiffQEC` module.
- `data.py` — DEM sample generation, parity split, curriculum sampler.
- `training.py` — training loop, EMA, checkpointing.
- `decoder.py` — `DiffQECDecoder` public API.
- `integrate.py` — pipeline glue (`decode_hardware_results_diffqec`).
- `smoke_test.py` — one-shot end-to-end runnable.
