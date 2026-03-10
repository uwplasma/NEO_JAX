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
- a synthetic one-surface ORBITS ``calc_cur = 1`` case used to validate
  ``neo_cur.*`` and ``current.dat`` against the real ``xneo`` executable

Legacy CLI parity
-----------------

The CLI regression coverage in ``tests/regression/test_cli_legacy.py`` runs the
reference executable from ``~/bin/xneo`` (or ``NEO_REFERENCE_BIN``) and compares
its outputs directly to the JAX CLI.

Current checks:

- exact text equality for:
  - ``neo_out.*``
  - ``neo_cur.*``
  - ``neolog.*``
  - ``diagnostic.dat``
  - ``diagnostic_add.dat``
  - ``diagnostic_bigint.dat``
  - ``conver.dat`` on the synthetic ORBITS parity case
- numerical equality, up to floating-point roundoff, for:
  - ``dimension.dat``
  - ``es_arr.dat``
  - all legacy ``*_arr.dat`` geometry dumps
  - ``current.dat`` token streams, including matching ``NaN`` / ``Infinity`` masks and tight tolerances on finite values
- control-file search-order parity for:
  - ``neo_param.<extension>``
  - ``neo_param.in``
  - ``neo_in.<extension>``
- slow-fixture CLI parity:
  - exact ``neo_out.*`` / ``neolog.*`` parity on ``ORBITS_FAST``
  - exact ``conver.dat`` columns 1-4 parity on ``ORBITS_FAST``
  - approximately ``rtol=5e-3`` parity on ``ncsx_c09r00_free_fast``
- optional GPU smoke parity:
  - CLI CPU-vs-GPU agreement on a one-surface ORBITS case
  - Python API CPU-vs-GPU agreement through ``run_neo(...)``

For the dense ``ORBITS_FAST`` case, NEO_JAX also exposes
``NEO_JAX_WRITE_IPMAX_DEBUG=1`` to emit ``diagnostic_ipmax_jax.dat``. That
debug trace is used to compare the per-step trapped-amplitude history against
the STELLOPT solver when investigating the remaining ``conver.dat`` fifth-column
discrepancy.

Supported legacy scope:

- ``calc_cur = 0`` parity is tested and supported
- ``calc_cur = 1`` parity is tested and supported

Precision
---------

NEO_JAX enables 64-bit JAX precision by default to match Fortran parity.
You can override this behavior by setting either:

- ``NEO_JAX_ENABLE_X64=0`` (NEO_JAX-specific)
- ``JAX_ENABLE_X64=0`` (global JAX setting)

Fast vs. full ORBITS parity:

- The default CLI regression suite runs the dense Landreman fixture plus
  reduced ORBITS / NCSX mini cases that finish quickly in CI.
- Full-fixture CLI parity checks for ``ORBITS_FAST`` and
  ``ncsx_c09r00_free_fast`` run when ``NEO_JAX_RUN_SLOW=1``.
- The separate full NCSX parity regression in
  ``tests/regression/test_ncsx_parity.py`` remains gated behind
  ``NEO_JAX_RUN_SLOW=1``.

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

GPU validation on ``office``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

NEO_JAX was revalidated on March 10, 2026 on the ``office`` workstation
(``pop-os``) with 2x NVIDIA RTX A4000 GPUs and JAX 0.6.2 in
``/home/rjorge/venvs/vmec_jax_gpu_bench``.

The GPU smoke suite is:

.. code-block:: bash

   env NEO_JAX_RUN_GPU=1 JAX_PLATFORM_NAME=gpu python -m pytest -q \
     tests/regression/test_gpu_smoke.py

That test file verifies:

- the legacy CLI produces the same one-surface ORBITS ``neo_out.*`` values on
  CPU and GPU
- the Python API produces the same ORBITS effective-ripple result on CPU and
  GPU
- the default CLI progress log reports the active JAX runtime

In addition, the user-facing ``examples/ncsx_epsilon_effective_plot.py`` script
was run on the same GPU host with ``MPLBACKEND=Agg`` and produced
``examples/ncsx_eps_eff_vs_s.png`` successfully.

Measured cold-run snapshots on ``office``:

.. list-table::
   :header-rows: 1

   * - Path
     - Case
     - CPU
     - GPU
     - CPU RSS MiB
     - GPU RSS MiB
   * - Legacy CLI
     - ``LandremanPaul2021_QA_lowres``
     - ``39.41 s``
     - ``95.71 s``
     - ``1908.4``
     - ``1966.4``
   * - Python API
     - ORBITS single-surface smoke
     - ``15.56 s first / 8.99 s reuse``
     - ``25.37 s first / 14.06 s reuse``
     - n/a
     - n/a

At the current problem sizes, the GPU path is functional and parity-checked but
still compile-bound. For the legacy CLI and the small ORBITS API smoke, the GPU
is slower than CPU because compile and launch overhead dominate the solve.
