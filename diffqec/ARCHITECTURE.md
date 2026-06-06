# DiffQEC Architecture Brief

## Package Layout

```
diffqec/
    __init__.py              — public exports
    diffusion.py             — discrete forward/reverse kernels, cosine schedule
    syndrome_processor.py    — spatial CNN + temporal GRU → syndrome feature c
    denoiser.py              — SFM layers, gated residual, bitwise logits
    model.py                 — top-level DiffQEC(nn.Module)
    data.py                  — DEM sample generation, parity split, curriculum sampler
    training.py              — training loop, EMA, checkpointing, curriculum schedule
    decoder.py               — public API: DiffQECDecoder
    integrate.py             — pipeline glue: decode_hardware_results_diffqec
    smoke_test.py            — one-shot end-to-end runnable
```

## Data Flow

1. **Training**
   - `data.py` generates DEM samples from a Stim circuit.
   - `ParityDataset` splits even/odd shots.
   - `CurriculumSampler` draws batches with round counts ∈ {3,5,7,9}.
   - `training.py` samples random timestep t, corrupts x0 → xt via `diffusion.py`,
     runs forward through `model.py`, computes bitwise CE loss.

2. **Inference**
   - `decoder.py` loads trained `DiffQEC`.
   - Syndrome dict from `extract_syndromes()` is reshaped/padded to (R, d, d) or (R, D).
   - Reverse diffusion: sample x_T ~ Bernoulli(0.5), iteratively denoise to x_0 = argmax(zθ).
   - `predicted_logical` and `confidence` returned.

3. **Pipeline Integration**
   - `integrate.py` wraps inference into `decode_hardware_results_diffqec(syndromes, model_path)`.
   - Returns `(ler, err)` lists per observable to satisfy existing contract.
   - `run_on_hardware.py` patched to accept `decoder="diffqec"|"mwpm"`.

## Key Design Decisions

- **Syndrome shape flexibility**: The rotated surface code does not produce a perfect R×d×d detector grid. The `SyndromeProcessor` accepts `(batch, rounds, D)` flattened syndromes and uses a 1D conv + GRU. This preserves the “spatial then temporal” paper structure while being robust to arbitrary detector counts. For codes where a clean 2D reshape is possible, a caller can pre-process.
- **Discrete end-to-end**: All tensors are float32 for PyTorch, but the binary state is preserved conceptually. The forward kernel uses `alpha_bar` and `beta` from the cosine schedule. The reverse sampler uses the closed-form posterior over binary states.
- **Confidence**: Per-shot confidence is the mean per-bit posterior probability of the predicted x0.
- **Curriculum**: Training begins with shortest round histories; every N epochs the max rounds increases by one step in {3,5,7,9}.
- **Checkpointing**: Saves `model.pt`, `ema_model.pt`, and `optimizer.pt` every epoch.

## Risk Register

| Risk | Mitigation |
|------|------------|
| Stim generated circuit shape mismatches syndrome processor | Use flattened 1D conv; reshape only when geometry is known |
| torch CPU training too slow for d=17 | Document that GPU is recommended; d=3/5 smoke tests stay fast |
| Parity split leakage | Enforce even/odd shot split at dataset level; never reshuffle |
| EMA desync | Save EMA shadow alongside model checkpoint |

## Acceptance Criteria

- [ ] `python -m diffqec.smoke_test` runs in < 60 s and prints LER < random baseline.
- [ ] `pytest tests/test_diffqec.py` passes (unit + integration).
- [ ] `decode_hardware_results_diffqec` returns finite `(ler, err)` on synthetic data.
- [ ] No modifications to `main` branch; all commits on `feature/diffqec-decoder`.
