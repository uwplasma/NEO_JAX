Validation
==========

NEO_JAX validation is organized into three layers:

- Geometry parity: Fourier reconstruction and spline evaluation match the
  Fortran outputs for ``R``, ``Z``, ``B``, and derived quantities.
- Integration parity: field-line integrals and trapped-particle sums reproduce
  the reference ``neo_out`` results on curated fixtures.
- End-to-end parity: the CLI output matches ``xneo`` within numerical tolerance.

Current reference cases include:

- ``ORBITS`` (tests/fixtures/orbits)
- ``NCSX`` tutorial example (tests/fixtures/ncsx)

Planned metrics:

- Relative error on ``epstot`` and ``epspar`` vs reference output.
- Consistency of derived quantities (``kg``, ``pard``, ``sqrt(g^{11})``).
- Regression of rational-surface handling and bin-averaged convergence.

Benchmarking
------------

The ``benchmarks/benchmark_orbits.py`` script measures runtime on the ORBITS
fixture using either the Python or JAX backend.
