Performance
===========

Fourier Summation Modes
-----------------------

NEO_JAX offers two Fourier evaluation paths:

- ``NEO_JAX_FOURIER_MODE=vectorized`` (default): fastest, but allocates
  ``theta × phi × mode`` temporaries.
- ``NEO_JAX_FOURIER_MODE=streamed``: avoids 3D temporaries by streaming over
  Fourier modes; reduces memory at the cost of additional runtime.

NCSX benchmark comparison (10 surfaces, CPU warmup run, ``/usr/bin/time -l``):

+------------+-----------+---------+
| Mode       | Total time| Max RSS |
+============+===========+=========+
| Vectorized | 51.37 s   | 4.45 GB |
+------------+-----------+---------+
| Streamed   | 58.78 s   | 2.55 GB |
+------------+-----------+---------+

Kernel Fusion Notes
-------------------

The JAX backend inlines the RHS evaluation, RK4 staging, and trapped-particle
updates inside the scan body. This keeps ``neo_eval`` and the RK4 stages in a
single fused region and reduces the number of separate kernels emitted by XLA.
The implementation lives in ``neo_jax.integrate.flint_bo_jax`` and replaces the
previous ``rk4_step`` + ``_process_trapped`` call boundary for the JIT path.

XLA Memory Hotspots (ORBITS_FAST)
---------------------------------

XLA memory reports for the ORBITS_FAST profile show the largest allocations in
``jit_flint_bo_jax`` are the spline coefficient arrays:

- ``b_spl``, ``g_spl``, ``k_spl``, ``p_spl`` with shape ``[4, 4, 25, 25]``.

In the scan body, the largest allocations are the per-step state and temporary
arrays with shape ``[25, 196]`` (theta × particle grids). These correspond to
RK4 staging and trapped-particle updates.

Profiling Workflow
------------------

Use ``benchmarks/profile_run.py`` to generate traces and XLA dumps:

.. code-block:: bash

   # Streamed Fourier profile
   NEO_JAX_FOURIER_MODE=streamed PYTHONPATH=. \
     python benchmarks/profile_run.py --jax --enable-x64 \
       --trace-dir profiles/trace_orbits_fast_streamed \
       --xla-dump-dir profiles/xla_orbits_fast_streamed

Open the trace directory with TensorBoard to inspect kernel-level hotspots.

For end-to-end VMEC→Boozer→NEO profiling (including the Boozer transform),
use ``benchmarks/profile_vmec_boozer_pipeline.py``. It records a JAX trace and
emits an HLO text dump for kernel-level inspection:

.. code-block:: bash

   python benchmarks/profile_vmec_boozer_pipeline.py \
     --case circular_tokamak \
     --trace-dir profiles/vmec_boozer_neo_trace \
     --hlo-out profiles/vmec_boozer_neo.hlo.txt

JIT Pipeline Reuse
------------------

For end-to-end VMEC→Boozer→NEO workflows, prefer
:func:`neo_jax.build_vmec_boozer_neo_jax` to precompute Boozer constants and
reuse a single compiled callable. This avoids recompiling the Boozer transform
and NEO scan in optimization loops.

Benchmark JIT reuse with:

.. code-block:: bash

   python benchmarks/benchmark_vmec_boozer_pipeline.py --case circular_tokamak --repeats 3

For CI, ``benchmarks/ci_perf_check.py`` provides a small regression guardrail
using a tiny pipeline case and configurable thresholds.
