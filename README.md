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

## Documentation

Sphinx documentation lives in `docs/` and is configured for Read the Docs.
See `docs/index.rst` for the table of contents.

## Status

This repository is under active development. See `PLAN.md` for the porting
plan and roadmap.
