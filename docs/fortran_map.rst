Fortran Mapping
===============

This page maps the original STELLOPT NEO Fortran routines to the JAX
implementation. It is the authoritative source for ensuring API parity and
output matching.

Main Program Flow
-----------------
- ``neo.f90`` -> ``neo_jax.cli`` and ``neo_jax.driver``
  - Reads control inputs.
  - Initializes global data.
  - Loops over flux surfaces.
  - Calls integration routine.

Initialization and Input
------------------------
- ``neo_read_control.f90`` -> ``neo_jax.control.read_control``
- ``read_booz_in.f90`` -> ``neo_jax.io.read_boozmn``
- ``neo_init.f90`` -> ``neo_jax.surface.init_surface``
- ``neo_prep.f90`` -> ``neo_jax.grids.prepare_grids``

Geometry and Fourier Sums
-------------------------
- ``neo_fourier.f90`` -> ``neo_jax.fourier.fourier_sums``
- ``neo_eval.f90`` -> ``neo_jax.geometry.neo_eval``
- ``neo_bderiv.f90`` -> ``neo_jax.geometry.neo_bderiv``

Surface Initialization
----------------------
- ``neo_init_s.f90`` -> ``neo_jax.surface.init_surface``
- ``neo_zeros2d.f90`` -> ``neo_jax.geometry.neo_zeros2d``

Integration
-----------
- ``flint_bo.f90`` -> ``neo_jax.integrate.flint_bo``
- ``rk4d_bo1.f90`` -> ``neo_jax.integrate.rk4_step``
- ``rhs_bo1.f90`` -> ``neo_jax.integrate.rhs_bo1``
- ``flint_bo.f90`` (scan backend) -> ``neo_jax.integrate.flint_bo_jax``

Splines
-------
- ``spl2d.f90`` -> ``neo_jax.splines.spl2d``
- ``eva2d.f90`` -> ``neo_jax.splines.eva2d``
- ``eva2d_fd.f90`` -> ``neo_jax.splines.eva2d_fd``
- ``eva2d_sd.f90`` -> ``neo_jax.splines.eva2d_sd``
- ``poi2d.f90`` -> ``neo_jax.splines.poi2d``

Output
------
- ``neo_output.f90`` -> ``neo_jax.cli`` output formatting
