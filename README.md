# NEO_JAX

JAX port of the STELLOPT NEO code for computing effective helical ripple and
related neoclassical transport diagnostics, with an end-to-end differentiable
pipeline through VMEC and Boozer transforms.

## Quick start

```bash
pip install -e .
```

```bash
neo-jax ORBITS --boozmn tests/fixtures/orbits/boozmn_ORBITS.nc --verbose
```

If you prefer not to install, run from the repo root with:

```bash
PYTHONPATH=. python examples/ncsx_jit_run.py
```

## Simple Python API

```python
from neo_jax import NeoConfig, run_boozmn

# Surfaces may be specified by index or by s in [0, 1].
config = NeoConfig(surfaces=[0.15, 0.35, 0.6, 0.85], theta_n=64, phi_n=64)
results = run_boozmn("boozmn.nc", config=config)

# Access by name
print(results.epsilon_effective)
print(results["epsilon_effective_by_class"].shape)
```

## Documentation

Sphinx documentation lives in `docs/` and is configured for Read the Docs.
See `docs/index.rst` for the table of contents.


## Examples

- `examples/ncsx_epsilon_effective_plot.py`: compute and plot epsilon effective vs `s`.
- `examples/ncsx_autodiff_rt0_optimization.py`: autodiff optimization demo over `rt0`.

## NCSX Parity Snapshot

![NCSX epstot parity](docs/assets/ncsx_epstot_compare.png)

| Metric | NEO (Fortran) | NEO_JAX (JAX) | Notes |
| --- | --- | --- | --- |
| Epsilon effective parity (max rel error, epstot) | — | 2.5e-10 | vs `tests/fixtures/ncsx/neo_out.ncsx_c09r00_free` |
| Runtime (10 surfaces, NCSX) | 60.37 s | 51.37 s | JAX time is steady-state after warmup |
| Max RSS (NCSX run) | 72.8 MiB | 4.45 GB | Measured via `/usr/bin/time -l` |

Repro commands:

```bash
# Fortran runtime + memory
/usr/bin/time -l /Users/rogerio/local/STELLOPT/NEO/Release/xneo ncsx_c09r00_free

# JAX runtime (steady-state) + memory
/usr/bin/time -l env PYTHONPATH=/Users/rogerio/local/tests/NEO_JAX \
  python /Users/rogerio/local/tests/NEO_JAX/benchmarks/benchmark_ncsx.py --jax --warmup
```


## Performance Tuning

NEO_JAX supports two Fourier evaluation modes:

- `NEO_JAX_FOURIER_MODE=vectorized` (default): fastest but allocates theta×phi×mode temporaries.
- `NEO_JAX_FOURIER_MODE=streamed`: lower memory by streaming over modes; slightly slower.

NCSX benchmark comparison (10 surfaces, CPU warmup run, `/usr/bin/time -l`):

| Mode | Total time | Max RSS |
| --- | --- | --- |
| Vectorized | 51.37 s | 4.45 GB |
| Streamed | 58.78 s | 2.55 GB |

## Status

This repository is under active development. See `PLAN.md` for the porting
plan and roadmap.

## Validation Cases

Current parity fixtures include:

- ORBITS (fast + full)
- NCSX tutorial case (fast by default; full gated by `NEO_JAX_RUN_SLOW=1`)
- LandremanPaul2021_QA_lowres (dense, 64x64 grid)
