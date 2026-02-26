Fortran Mapping
===============

This page maps the original STELLOPT NEO Fortran routines to the intended JAX
functions and modules. It is the authoritative source for ensuring API parity
and output matching.

Main Program Flow
-----------------
- `neo.f90` -> `neo_jax.run.main`
  - Reads control inputs.
  - Initializes global data.
  - Loops over flux surfaces.
  - Calls integration routine.

Initialization and Input
------------------------
- `neo_read_control.f90` -> `neo_jax.io.read_control`
- `neo_read.f90` -> `neo_jax.io.read_boozer` (file compatibility mode only)
- `neo_init.f90` -> `neo_jax.init.init_global`
- `neo_prep.f90` -> `neo_jax.init.prepare_grids`

Geometry and Fourier Sums
-------------------------
- `neo_fourier.f90` -> `neo_jax.geometry.fourier_sums`
- `neo_eval.f90` -> `neo_jax.geometry.eval_splines`

Surface Initialization
----------------------
- `neo_init_s.f90` -> `neo_jax.init.init_surface`
- `neo_zeros2d.f90` -> `neo_jax.math.root_find_2d`

Integration
-----------
- `flint_bo.f90` -> `neo_jax.integrate.flint_bo`
- `rk4d_bo1.f90` -> `neo_jax.integrate.rk4_step`
- `rhs_bo1.f90` -> `neo_jax.integrate.rhs_bo1`

Splines
-------
- `spl2d.f90` -> `neo_jax.splines.spl2d`
- `eva2d.f90` -> `neo_jax.splines.eva2d`
- `eva2d_fd.f90` -> `neo_jax.splines.eva2d_fd`
- `eva2d_sd.f90` -> `neo_jax.splines.eva2d_sd`
- `poi2d.f90` -> `neo_jax.splines.poi2d`

Output
------
- `neo_output.f90` -> `neo_jax.io.write_output`
