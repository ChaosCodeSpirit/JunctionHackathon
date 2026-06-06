# Belief-Propagation Decoder

Belief propagation on the surface-code detector graph, via
[`ldpc`](https://github.com/quantumgizmos/ldpc). Mirrors the public
interface of the `mwpm` package so the two decoders can be swapped
through `decode_hardware_results(decoder=...)`.

## Quick start

```bash
python -m bp.smoke_test
```

## Decode

```python
from bp import build_bp_decoder, decode_hardware_results_bp

decoder, O = build_bp_decoder(circuit)        # BP+OSD, order 2
ler, err, confidence = decode_hardware_results_bp(syndromes, decoder=decoder, O=O)
```

Or through the pipeline dispatcher:

```python
from run_on_hardware import decode_hardware_results

ler, err = decode_hardware_results(
    syndromes, decoder="bp", stim_circuit_noisy=circuit,
)
```

## What this actually is: BP+OSD, not plain BP

By default, `build_bp_decoder` uses `ldpc.BpOsdDecoder` with
`osd_order=2`. This is the practical working point for QEC, and
it is **not** the same as plain loopy BP.

- **Plain BP** (`osd_order=0`): vanilla loopy belief propagation. On
  surface codes, it converges to a stable but sub-optimal fixed
  point because of short cycles in the factor graph. Empirically 3-4×
  worse than MWPM at d=3 / p=0.001. Useful as a baseline and for
  convergence studies; not recommended as the final decoder.
- **BP+OSD** (default): BP first, then ordered statistics decoding
  post-processes the BP result. At `osd_order≥2` it matches MWPM
  exactly at d=3 / p=0.001.

The reason BP alone loses to MWPM on surface codes is well-known
in the QEC literature — short cycles trap the message-passing
dynamics. The Yao et al. (2023) paper on this exact issue
([arXiv:2312.10950](https://arxiv.org/abs/2312.10950)) is about
qLDPC codes, where the same problem is much worse; for surface
codes BP+OSD is the standard fix.

## Why BP+OSD and not just MWPM

Same factor graph (built from the same stim DEM), but:

- **Soft outputs** — `confidence[i, j]` is `P(observable_j = predicted_j | syndrome_i)`,
  derived from the LLR of the BP result projected onto the observable.
  This is what `mwpm`'s MWPM branch can't give you.
- **Calibration integration** — replace the DEM-derived priors with
  per-edge weights from Resonance calibration data; the rest of the
  algorithm is unchanged.
- **Strictly more expressive than MWPM** near threshold or under
  model mismatch.

For the hackathon at d=3 / 1 round / p=0.001 (deep sub-threshold),
the LER is statistically indistinguishable from MWPM. The win is
the soft output and the analysis it enables (hard-shot
identification, spatial correlation with chip regions,
low-confidence filtering on LER aggregation).

## On the diffqec mismatch

Like `mwpm/`, this package does not mirror `diffqec/`'s full file
layout — there is no model, no training, no data.py. Just functions.

| | mwpm | bp |
|---|---|---|
| Algorithm | min-weight matching | BP+OSD on the same factor graph |
| Output | hard | soft (LLR-derived `P(obs)`) |
| Library | `pymatching` | `ldpc` |
| LER at d=3, p=0.001 | 0.0012 | 0.0012 (with OSD) |
| Engineering | trivial | small (DEM parser + ldpc) |

## Files

- `integrate.py` — `dem_to_pcm`, `build_bp_decoder`, `decode_hardware_results_bp`.
- `smoke_test.py` — one-shot end-to-end runnable with BP-vs-MWPM comparison.

## Limitations

- **Plain BP underperforms MWPM on surface codes.** Use BP+OSD.
  Set `osd_order=0` only if you want to study BP's failure modes.
- **Single-shot decoding loop.** ldpc 2.4 has no batched API, so
  wall time scales linearly with shot count. For LUMI-scale sweeps,
  parallelise the outer loop with joblib/multiprocessing.
- **Soft confidence is an indicative ranking**, not a calibrated
  posterior marginal. Use it to identify "hard shots" (low
  confidence) and correlate with hardware, not as a frequentist
  confidence interval.
- **No damping / no convergence-based fall-back to MWPM yet.**
  Could be added by checking `decoder.converge` after each shot
  and falling back to PyMatching on non-converged cases.
