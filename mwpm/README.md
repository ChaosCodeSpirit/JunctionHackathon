# MWPM Decoder

Minimum-weight perfect matching on the surface-code detector graph.
PyMatching does the work; this package is a thin pipeline shim so
`mwpm` and `diffqec` can be swapped through the same
`decode_hardware_results(decoder=...)` entry point.

## Quick start

```bash
python -m mwpm.smoke_test
```

## Decode

```python
from mwpm import build_matching_decoder, decode_hardware_results_mwpm

matcher = build_matching_decoder(noisy_circuit)   # build once, amortize
ler, err = decode_hardware_results_mwpm(syndromes, matcher=matcher)
```

Or through the pipeline dispatcher:

```python
from run_on_hardware import decode_hardware_results

ler, err = decode_hardware_results(syndromes, decoder="mwpm",
                                    stim_circuit_noisy=circuit)
```

## On the diffqec mismatch

`diffqec/` has a class-based API, training data helpers, and a soft
confidence return. MWPM has none of those — the matching graph is
closed-form and PyMatching already exposes the state. Mirroring the
`diffqec` file layout would add empty scaffolding, so `mwpm/` is just
two functions in one module:

| diffqec | mwpm | why |
|---|---|---|
| `DiffQECDecoder` class | `decode_hardware_results_mwpm()` function | no stateful model |
| `confidence` per shot | not returned | MWPM is hard-decision |
| `data.py` (training data) | — | no training |
| `model.py`, `training.py` | — | closed-form |

`build_matching_decoder` is the only expensive call; reuse the
returned `Matching` across batches.
