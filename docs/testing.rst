Testing and CI
==============

NEO_JAX is tested at several levels: low-level numerics, solver behavior on
reference fixtures, CLI behavior, performance guardrails, and optional GPU
smoke tests.

Testing layers
--------------

The test suite is organized around complementary questions:

- do low-level spline, Fourier, and I/O routines produce the expected arrays?
- do solver outputs remain stable on curated reference cases?
- do the user-facing Python API and CLI behave as documented?
- do optional GPU and performance checks stay within acceptable bounds?

Representative test files include:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Test file
     - Coverage
   * - ``tests/unit/test_api.py``
     - Public API behavior, result accessors, surface mapping, and JAX-path
       return types.
   * - ``tests/unit/test_control.py``
     - Control-file parsing.
   * - ``tests/regression/test_constellaration_guard.py``
     - Low-``|iota|`` safeguards and approximate fallback behavior.
   * - ``tests/regression/test_cli_legacy.py``
     - CLI file generation, progress logging, and parity against committed
       ``xneo`` reference fixtures for the file-based workflow.
   * - ``tests/regression/test_landreman_qa_lowres_parity.py``
     - Dense QA fixture comparison.
   * - ``tests/regression/test_orbits_parity.py``
     - ORBITS reference behavior.
   * - ``tests/regression/test_ncsx_parity.py``
     - NCSX comparison case.
   * - ``tests/regression/test_gpu_smoke.py``
     - Optional CPU-versus-GPU agreement checks.

Local test commands
-------------------

Common local validation commands are:

.. code-block:: bash

   pytest -q tests/unit/test_api.py
   pytest -q tests/regression/test_constellaration_guard.py
   pytest -q tests/regression/test_cli_legacy.py
   pytest -q tests/regression/test_landreman_qa_lowres_parity.py
   pytest -q tests/regression/test_orbits_parity.py
   pytest -q tests/regression/test_ncsx_parity.py

The documentation build is also part of the release workflow:

.. code-block:: bash

   python -m sphinx -b dummy docs docs/_build/dummy

Continuous integration
----------------------

The repository CI runs on GitHub Actions. The workflow installs the package,
pulls the VMEC and Boozer dependencies used by the pipeline tests, runs the
full pytest suite on CPU, and executes a small performance regression check.

.. literalinclude:: ../.github/workflows/ci.yml
   :language: yaml

What the CI does not do by default
----------------------------------

Some checks are intentionally opt-in:

- full slow reference cases behind ``NEO_JAX_RUN_SLOW=1``
- GPU smoke tests behind ``NEO_JAX_RUN_GPU=1``
- external NCSX fixture consumers behind ``NEO_JAX_FETCH_EXTERNAL_FIXTURES=1``

That separation keeps standard CI fast while still preserving the heavier
validation workflows for release and benchmarking.

Performance guardrails
----------------------

The CI workflow also runs ``benchmarks/ci_perf_check.py`` with explicit limits
on compile and reuse times. This is not a substitute for full benchmarking, but
it catches large regressions in the compiled JAX path early.

Reference comparisons
---------------------

When NEO_JAX is compared against established external reference outputs, the
goal is to verify correctness of the current implementation on representative
geometries. Those comparisons are summarized in :doc:`validation`. The default
CLI parity suite uses committed ``xneo`` reference fixtures so the checks stay
fully reproducible on CI and on developer machines without a local STELLOPT
build. The testing language in NEO_JAX is therefore evidence-driven:
reference agreement is part of the acceptance story, while the user-facing
solver remains NEO_JAX itself.
