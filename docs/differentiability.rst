Differentiability
=================

NEO_JAX is designed for end-to-end automatic differentiation when coupled with
JAX-native equilibrium and Boozer transforms. :cite:`vmec-jax,booz-xform-jax,jax-docs`

Key design choices
------------------

The implementation exposes JAX where it matters most:

- Pure JAX array operations for spline evaluation and ODE right-hand-sides.
- A scan-based integrator (``flint_bo_jax``) that keeps the time-stepping loop
  on-device for JIT compilation.
- Static configuration bundles (grid sizes, mode limits) to avoid recompilation.
- A JAX-native surface-scan path for repeated multi-surface execution.

Limitations and roadmap
-----------------------

The JAX backend includes the same main transport logic as the Python solver,
including the rational-surface correction path when applicable. For
differentiable workflows, keep file-output and debugging callbacks disabled,
since they introduce host-side effects.

For performance-critical workflows, we recommend:

- JIT-compiling the scan backend over batches of surfaces.
- Using 64-bit precision when matching Fortran outputs.
- Keeping diagnostic prints off during JIT execution.

Reverse-mode autodiff (``jax.grad``) through the scan is currently limited by
dynamic loop bounds in the trapped-particle logic. Forward-mode
(``jax.jvp``/``jax.jacfwd``) is supported and used in the optimization examples.

What differentiates cleanly
---------------------------

The cleanest differentiated paths are:

- geometry-to-output calculations that stay in the JAX backend
- repeated solves built with :func:`neo_jax.build_vmec_boozer_neo_jax`
- parameter studies using forward-mode sensitivities

Practical guidance:

- use ``jax_surface_scan=True`` when the solve should stay on device
- reuse compiled callables rather than rebuilding geometry in every outer step
- prefer forward-mode when the number of design variables is modest
- disable diagnostic dumps in differentiable runs

JAX-native Boozer transforms
----------------------------

For a fully differentiable pipeline, pair NEO_JAX with the functional API
in ``booz_xform_jax.jax_api``. This avoids Python loops over surfaces and keeps
all arrays on-device for JIT and ``jax.grad``.

For repeated solves, use :func:`neo_jax.build_vmec_boozer_neo_jax` to build a
reusable, optionally JIT-compiled callable that closes over the static Boozer
constants and grid setup.
