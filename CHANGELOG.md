# Changelog

## v1.0.1 - 2026-04-18

Packaging and release-preparation update.

Highlights

- Removed version pins from runtime, development, and documentation dependencies.
- Expanded GitHub Actions CI to Python 3.10, 3.11, and 3.12.
- Reworked CLI parity tests to use committed ``xneo`` reference fixtures, so
  parity is checked without needing a local STELLOPT build.
- Added installed-package CLI smoke coverage in CI.
- Added a docs build job to GitHub Actions.
- Added a trusted-publishing workflow for PyPI releases.
- Documented direct PyPI installation with ``pip install neo-jax``.

## v1.0.0 - 2026-04-17

NEO_JAX 1.0.0 is the first full GitHub release of the code as a standalone
solver for effective ripple and related trapped-particle diagnostics in Boozer
coordinates.

Highlights

- Standalone Python and JAX solver for `epsilon_effective` and related
  diagnostics.
- Clean high-level API built around `NeoConfig`, `run_neo`, `NeoResults`, and
  direct support for in-memory Boozer objects.
- End-to-end VMEC→Boozer→NEO pipeline helpers through `vmec_jax` and
  `booz_xform_jax`.
- JAX scan backend, streamed/vectorized Fourier modes, GPU smoke coverage, and
  reusable compiled pipeline entrypoints.
- File-based CLI workflow with `xneo`, `xneo_jax`, and `python -m neo_jax`.
- Low-`|iota|` preflight safeguards plus an opt-in approximate fallback for
  pathological rational-surface workloads.
- Expanded scientific documentation covering theory, numerics, geometry
  loading, runtime controls, testing, and source structure.
- Hardened CI so optional VMEC example-data dependencies skip cleanly when they
  are not present in the GitHub Actions environment.

Release status

- Current release tag: `v1.0.0`
- Package version: `1.0.0`
- Default branch: `main`
- GitHub Actions CI: passing on the release commit
