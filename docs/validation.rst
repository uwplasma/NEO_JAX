Validation
==========

NEO_JAX validation is organized into three layers:

- Geometry parity: Fourier reconstruction and spline evaluation match the
  Fortran outputs for ``R``, ``Z``, ``B``, and derived quantities.
- Integration parity: field-line integrals and trapped-particle sums reproduce
  the reference ``neo_out`` results on curated fixtures.
- End-to-end parity: the CLI output matches ``xneo`` file-by-file on the
  supported legacy cases.

Current reference cases include:

- ``ORBITS`` (tests/fixtures/orbits)
- ``NCSX`` tutorial example (tests/fixtures/ncsx)
- ``LandremanPaul2021_QA_lowres`` (tests/fixtures/landreman_qa_lowres)
- a synthetic one-surface ORBITS legacy case used to validate
  ``neo_out.*``, ``neolog.*``, ``diagnostic*.dat``, ``conver.dat``, and the
  legacy ``*_arr.dat`` dumps against the real ``xneo`` executable

Legacy CLI parity
-----------------

The CLI regression coverage in ``tests/regression/test_cli_legacy.py`` runs the
reference executable from ``~/bin/xneo`` (or ``NEO_REFERENCE_BIN``) and compares
its outputs directly to the JAX CLI.

Current checks:

- exact text equality for:
  - ``neo_out.*``
  - ``neolog.*``
  - ``diagnostic.dat``
  - ``diagnostic_add.dat``
  - ``diagnostic_bigint.dat``
  - ``conver.dat``
- numerical equality, up to floating-point roundoff, for:
  - ``dimension.dat``
  - ``es_arr.dat``
  - all legacy ``*_arr.dat`` geometry dumps

Supported legacy scope:

- ``calc_cur = 0`` parity is tested and supported
- ``calc_cur = 1`` is not yet ported in the CLI layer

Precision
---------

NEO_JAX enables 64-bit JAX precision by default to match Fortran parity.
You can override this behavior by setting either:

- ``NEO_JAX_ENABLE_X64=0`` (NEO_JAX-specific)
- ``JAX_ENABLE_X64=0`` (global JAX setting)

Fast vs. full ORBITS parity:

- The default regression test uses the reduced ``ORBITS_FAST`` fixture
  (2 surfaces, 25x25 grid) for quick CI runs.
- Set ``NEO_JAX_ORBITS_FULL=1`` to run the full ORBITS parity test overnight.
- ``neo_in.ncsx_c09r00_free_fast`` provides a reduced NCSX parity test that runs
  by default in CI (4 surfaces, 64x64 grid).
- The full NCSX parity regression remains gated behind ``NEO_JAX_RUN_SLOW=1`` to
  avoid running the slow-path integrator in standard CI.

Planned metrics:

- Relative error on ``epstot`` and ``epspar`` vs reference output.
- Consistency of derived quantities (``kg``, ``pard``, ``sqrt(g^{11})``).
- Regression of rational-surface handling and bin-averaged convergence.

Benchmarking
------------

The ``benchmarks/benchmark_orbits.py`` script measures runtime on the ORBITS
fixture using either the Python or JAX backend.

For JAX performance runs, the driver uses a JIT-compiled kernel by default.
Set ``NEO_JAX_DISABLE_JIT=1`` to force eager execution when debugging.

Both benchmark scripts accept ``--warmup`` to separate compile time from
steady-state runtime.

Profiling
~~~~~~~~~

Use ``benchmarks/profile_run.py`` to capture JAX traces and XLA dumps for
performance and memory analysis:

.. code-block:: bash

   # Fast ORBITS trace with X64 parity settings
   PYTHONPATH=. python benchmarks/profile_run.py --jax --enable-x64 \
     --trace-dir profiles/orbits_fast

   # Dump XLA HLO/LLVM artifacts (CPU or GPU)
   PYTHONPATH=. python benchmarks/profile_run.py --jax --enable-x64 \
     --xla-dump-dir profiles/xla_orbits_fast

The trace directory can be opened with TensorBoard:

.. code-block:: bash

   tensorboard --logdir profiles

GPU Run Guide
~~~~~~~~~~~~~

To benchmark on GPU, ensure a CUDA-enabled JAX build is installed and set the
runtime environment before running the benchmark scripts:

.. code-block:: bash

   export JAX_PLATFORM_NAME=gpu
   export JAX_ENABLE_X64=1
   export XLA_PYTHON_CLIENT_PREALLOCATE=false
   export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

   python benchmarks/benchmark_orbits.py --jax
   python benchmarks/benchmark_ncsx.py --jax

Tips:

- Use ``JAX_ENABLE_X64=1`` to match Fortran parity expectations.
- If you see out-of-memory errors, lower ``XLA_PYTHON_CLIENT_MEM_FRACTION`` or
  set ``XLA_PYTHON_CLIENT_PREALLOCATE=false``.
