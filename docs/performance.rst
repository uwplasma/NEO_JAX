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
