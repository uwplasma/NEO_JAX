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

The JAX backend includes the rational-surface correction logic matching the
reference Fortran implementation. For end-to-end differentiation, keep
diagnostic file dumps disabled (they trigger host callbacks).

For performance-critical workflows, we recommend:

- JIT-compiling the scan backend over batches of surfaces.
- Using 64-bit precision when matching Fortran outputs.
- Keeping diagnostic prints off during JIT execution.

Reverse-mode autodiff (``jax.grad``) through the scan is currently limited by
dynamic loop bounds in the trapped-particle logic. Forward-mode
(``jax.jvp``/``jax.jacfwd``) is supported and used in the optimization examples.

JAX-native Boozer transforms
----------------------------

For a fully differentiable pipeline, pair NEO_JAX with the functional API
in ``booz_xform_jax.jax_api``. This avoids Python loops over surfaces and keeps
all arrays on-device for JIT and ``jax.grad``.

For repeated solves, use :func:`neo_jax.build_vmec_boozer_neo_jax` to build a
reusable, optionally JIT-compiled callable that closes over the static Boozer
constants and grid setup.
