Differentiability
=================

NEO_JAX is designed for end-to-end automatic differentiation when coupled with
JAX-native equilibrium and Boozer transforms. :cite:`vmec-jax,booz-xform-jax`

Key design choices include:

- Pure JAX array operations for spline evaluation and ODE right-hand-sides.
- A scan-based integrator (``flint_bo_jax``) that keeps the time-stepping loop
  on-device for JIT compilation.
- Static configuration bundles (grid sizes, mode limits) to avoid recompilation.

Limitations and roadmap
-----------------------

The current JAX backend does not yet implement the rational-surface correction
logic found in the reference Fortran code. When high fidelity near rational
surfaces is required, use the Python-loop backend (``flint_bo``) or ensure
``calc_nstep_max`` is set to disable the correction.

For performance-critical workflows, we recommend:

- JIT-compiling the scan backend over batches of surfaces.
- Using 64-bit precision when matching Fortran outputs.
- Keeping diagnostic prints off during JIT execution.
