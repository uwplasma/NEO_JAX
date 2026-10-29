# Changelog

## v1.1.1 - 2026-10-23

- Keep rational-correction periods positive for negative iota while preserving
  signed field-line winding (#17).
- Correct NFP scaling in the rational-surface work guard (#21).
- Handle exact rational closure and reject invalid or overflowing correction
  counts before integer conversion (#22).

## v1.1.0 - 2026-10-18

- booz_xform_jax>=0.5.0 is now a core dependency. boozmn files are read with
  its shared reader, so NEO_JAX and VMEX parse them identically (#19).
- Effective ripple is unchanged: on three decks (including a LASYM-padded
  one) results are bit-identical to 1.0.6, with no slowdown.

## v1.0.6 - 2026-10-15

- Keep cosine and sine Boozer spectra through the readers, the booz_xform_jax
  adapter and the VMEX pipeline, so non-stellarator-symmetric (LASYM) equilibria
  give an effective ripple. Incomplete sine families are rejected (#7).
- One asymmetric case matches STELLOPT NEO to 1.3e-10 relative error; zero sine
  tables reproduce the symmetric path exactly. See docs/validation.rst.

## v1.0.5 - 2026-10-02

- Faster CPU effective-ripple kernels with per-surface stopping.
- Documented cold and warm timings for a callback-free QA profile.

## v1.0.4 - 2026-10-02

- Reduce cold spline compilation and host mapping setup without changing
  integration controls.
- Align packed spectra with radial iota and current profiles; reject ambiguous
  metadata.
- Require booz_xform_jax 0.4.3 for the VMEX pipeline.

## v1.0.3 - 2026-10-01

- Reuse compiled surface kernels across surfaces and coefficient updates.
- Use the maintained VMEX pipeline with booz_xform_jax 0.4.2 or newer.
- Keep axisymmetric extrema finite and reject unsupported asymmetric NetCDF geometry.
- Refresh Fortran accuracy comparisons, timings, examples and capability documentation.

## v1.0.2 - 2026-05-25

Slim-package and fixture-distribution update.

Highlights

- Removed the large NCSX Boozer NetCDF and the uncompressed legacy
  ``current.dat`` reference from the repository history.
- Added an external-fixture resolver so opt-in NCSX examples and regression
  tests fetch the large Boozer file from a GitHub release asset cache instead
  of bundling it into normal clones or package artifacts.
- Kept GitHub Actions coverage for the NCSX path by enabling explicit fixture
  fetches in CI, while retaining slim default behavior for local checkouts.
- Guarded the PyPI release workflow so auxiliary fixture-only releases do not
  fail packaging automation.
- Reduced the source and wheel distributions to roughly 64 KB with no large
  test fixtures included.

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
