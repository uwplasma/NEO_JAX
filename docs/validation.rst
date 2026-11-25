Validation
==========

NEO_JAX validation is organized into three layers:

- Geometry parity: Fourier reconstruction and spline evaluation match the
  Fortran outputs for ``R``, ``Z``, ``B``, and derived quantities.
- Integration parity: field-line integrals and trapped-particle sums reproduce
  the reference ``neo_out`` results on curated fixtures.
- End-to-end parity: the CLI output matches ``xneo`` file-by-file on the
  supported legacy cases, using committed reference fixtures generated with
  ``xneo``.

Current reference cases include:

- ``ORBITS`` (tests/fixtures/orbits)
- ``NCSX`` tutorial example (tests/fixtures/ncsx)
- ``LandremanPaul2021_QA_lowres`` (tests/fixtures/landreman_qa_lowres)
- ``constellaration`` low-``|iota|`` fixtures
  (tests/fixtures/constellaration)
- a synthetic one-surface ORBITS legacy case used to validate
  ``neo_out.*``, ``neolog.*``, ``diagnostic*.dat``, ``conver.dat``, and the
  legacy ``*_arr.dat`` dumps against stored ``xneo`` reference outputs
- a synthetic one-surface ORBITS ``calc_cur = 1`` case used to validate
  ``neo_cur.*`` and ``current.dat`` against stored ``xneo`` reference outputs
- a synthetic one-surface NCSX case used to validate ``neo_out.*`` and
  ``neolog.*`` against stored ``xneo`` reference outputs

The large NCSX Boozer input is distributed as an external release asset rather
than being committed directly in the repository. Default CI and slim checkouts
therefore skip NCSX fixture consumers unless
``NEO_JAX_FETCH_EXTERNAL_FIXTURES=1`` is set or the file is already present in
the local fixture cache.

Asymmetric geometry
-------------------

Tests retain all four complementary coefficient families through file readers,
object adapters and the differentiable VMEX pipeline. Analytic Fourier values,
angular derivatives and end-to-end derivatives are checked independently.

A Landreman–Sengupta–Plunk stellarator surface agrees with corrected STELLOPT
NEO to ``1.3e-10`` relative effective-ripple error. Three other asymmetric
reference surfaces returned NaNs and do not establish ripple parity.
Two ORBITS surfaces with shifted coordinates agree within ``1e-10``;
removing sine ``|B|`` changes the ripple.

Zero sine tables on the asymmetric path reproduce the symmetric path bit for
bit. Flipping every sine partner (the mirror image) changes the ripple by
``1e-3`` to ``5e-3`` relative, which is the finite field-line length error and
falls with ``acc_req``. For an exact up-down asymmetric Solov'ev equilibrium,
an independent real-space evaluation of Nemov et al. (1999) gives ``9e-16``,
as axisymmetry requires; NEO_JAX on the converged VMEX solution gives
``4e-6`` to ``1.4e-5`` at s = 0.25 to 0.8. That value is independent of Boozer
and angular resolution and persists with the sine tables removed, so it is a floor of the
NEO algorithm, not of the asymmetric path.


Legacy CLI parity
-----------------

The CLI regression coverage in ``tests/regression/test_cli_legacy.py`` runs the
JAX CLI against frozen reference outputs committed under
``tests/fixtures/cli_legacy/``. Those files were generated once with the
STELLOPT ``xneo`` executable and then checked into the repository so the tests
do not require an external binary.

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

Low-``|iota|`` regression coverage
----------------------------------

The ``constellaration`` fixtures contain surfaces with very small rotational
transform. For these cases, the legacy rational-surface correction implies

.. code-block:: text

   nfp_rat ~= ceil(1 / acc_req / |iota|)

so the required field-period count can become enormous. Physically, this is the
near-zero-``|iota|`` regime. Numerically, it can make the legacy correction
look hung simply because the requested work is very large.

``tests/regression/test_constellaration_guard.py`` now verifies that:

- both reported Boozer files fail fast instead of hanging
- the exception message explains that the requested rational correction is too large
- ``progress=True`` prints the detailed preflight diagnostic before aborting
- the same safeguard applies to the JAX surface-scan backend
- ``rational_surface_policy="approximate"`` returns a finite controlled result
  with explicit approximation diagnostics
- approximate-mode JAX and Python per-surface backends stay close on the same
  low-resolution pathological case
- requesting ``jax_surface_scan=True`` with approximate mode falls back to the
  per-surface path instead of trying to support approximation inside the scan
  backend

The broader parity suites for ORBITS, NCSX, LandremanPaul2021_QA_lowres, the
legacy CLI outputs, and the public API continue to exercise the unchanged
paths. Approximate mode is opt-in, and the default safeguard does not alter
those existing cases unless the preflight rational-work estimate actually
exceeds the configured limit.

Operational guidance:

- keep the default safeguard for validation and parity work
- use ``rational_surface_policy="approximate"`` when a controlled fallback is
  more useful than an early error
- set ``max_rational_field_periods=0`` only when you explicitly want the full
  exact legacy correction, even if it may take hours
- avoid surfaces with ``|iota|`` near zero, or loosen ``acc_req``, when
  scanning new equilibria

Precision
---------

NEO_JAX enables 64-bit JAX precision by default to match Fortran parity.
You can override this behavior by setting either:

- ``NEO_JAX_ENABLE_X64=0`` (NEO_JAX-specific)
- ``JAX_ENABLE_X64=0`` (global JAX setting)

Fast vs. full ORBITS parity:

- The default CLI regression suite runs the dense Landreman fixture plus
  reduced ORBITS / NCSX mini cases that finish quickly in CI.
- Full solver parity checks for dense ORBITS and NCSX fixtures remain available
  behind ``NEO_JAX_RUN_SLOW=1`` in the public-API regression tests.

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

GPU validation
~~~~~~~~~~~~~~

NEO_JAX was validated on March 10, 2026 with two NVIDIA RTX A4000 GPUs, Linux and JAX 0.6.2.

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

Measured cold-run snapshots:

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

Fast-ion proxies
----------------

:func:`neo_jax.fast_ion_proxies` is checked in ``tests/test_fastion.py``:

- the drift integrals of one well against finite differences of the action
  :math:`J` computed independently with scipy: 1e-4 for
  :math:`\partial_\alpha J`, 1e-3 for :math:`\partial_s J`;
- a single helical harmonic (omnigenous): :math:`\Gamma_c` = 2.4e-4 and
  :math:`\Gamma_\alpha` = 1.4e-3, against 1e-2 and 0.2 for a rippled field;
- :math:`\Gamma_\alpha` never exceeds the trapped fraction;
- forward mode against central differences for both proxies, and a bound on
  the second differences of :math:`\Gamma_\alpha` along a scan.

Through the VMEX equilibrium solve (reverse mode against re-solved central
differences, four boundary coefficients): :math:`\Gamma_c` within 0.05-0.3% on
an nfp = 2 seed and 0.1-7% on QA, QI and nfp = 3 cases;
:math:`\Gamma_\alpha` within 5% on QA and within 1% on the large components
of the nfp = 2 seed. A quasi-helical case shows a factor 3 between the
:math:`\Gamma_c` gradient through the solve and finite differences while the
kernel derivative at fixed Boozer spectrum agrees to 1e-3; the cause is
upstream of this module and is open.

On the analytic scan of ``examples/fast_ion_proxies.py`` the forward-mode
derivative of :math:`\Gamma_\alpha` agrees with central differences to 0.4% of
its largest value. That of :math:`\Gamma_c` agrees to 3-25% depending on the
point and the step, and the slope at single points changes by up to a factor 2
between 16 and 64 lines while the value changes by 1%: the integrand has a cusp
where the poloidal drift vanishes, which equispaced field lines resolve slowly
in the derivative.

Convergence with field lines (16, 32, 64) at s = 0.5: :math:`\Gamma_c` within
1% on QA, QH and QI and 5% on an nfp = 4 ellipse with iota near a low-order
rational; :math:`\Gamma_\alpha` within 0.2% on QA and 1.6% per doubling on QI.

Smoothed against sharp :math:`\Gamma_\alpha` (threshold half-width 0.2
against a Heaviside): -0.4% on QA, -9% on the nfp = 4 ellipse, +29% on QH and
+59% on QI, where many wells sit just below the threshold. No published
:math:`\Gamma_c` or :math:`\Gamma_\alpha` value is reproduced here. DESC's
``Gamma_c`` uses Nemov's pointwise form and is a different quantity.

Bounce-integral effective ripple
--------------------------------

:func:`neo_jax.effective_ripple_3_2` is checked in ``tests/unit/test_ripple.py``:

- a single helicity (QA, QH, with or without current) gives zero, a perturbation scales quadratically, and the
  value is invariant under Boozer angle shifts and field scaling;
- Fortran ``xneo`` and the field-line integrator (both at 128 x 128 spline, 96 particles, 80 steps per period,
  ``acc_req = 1e-3``) on an nfp = 4 quasi-isodynamic spectrum (finite beta, iota about -1.3):
  5.19e-4, 4.79e-4 against 5.20e-4, 4.79e-4 for the bounce integrals at a 3x budget; the default budget is within 1%;
- a vacuum nfp = 2 ellipse: ``xneo`` 1.18e-2, 1.67e-2, 2.15e-2 and DESC (converged bounce resolution)
  1.21e-2, 1.69e-2, 2.18e-2 against 1.20e-2, 1.66e-2, 2.13e-2;
- an independent numpy evaluation (one 150-turn field line, linear tables, 800 pitch points) gives 4.784e-4 on the QI
  surface where the bounce code gives 4.787e-4;
- the true QA deck ``LandremanPaul2021_QA_lowres`` has an effective ripple^(3/2) of about 1e-8, where the
  default-resolution field-line integrator returns spline noise; the bounce code, a fine ``xneo`` run and the
  budget-converged value agree within 5%.

DESC returns 3.0 to 3.1e-4 (against 4.7 to 4.8e-4) on the QI spectrum at every resolution and at zero, half and
full pressure. DESC takes the geodesic curvature from the geometry of the equilibrium it fits to the wout, which has
a scaled force residual of 0.013 there; the grad-B form on the same DESC data gives 4.6e-4. The two differ by 8% rms
pointwise, and the per-well geodesic integral cancels inside a QI well, so the ripple moves by 47%. After re-solving
in DESC (residual 1.25e-4) its two forms agree to 4%. The same sensitivity applies to the equilibrium solve here:
on this deck a force tolerance of 1e-10 gives +8% and 1e-12 gives +0.3%, so near-omnigenous cases need 1e-12.
