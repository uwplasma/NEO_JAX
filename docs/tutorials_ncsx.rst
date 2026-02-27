Tutorial: NCSX Example
======================

This example mirrors the NEO tutorial case based on the NCSX equilibrium. The
input files are in ``tests/fixtures/ncsx`` and correspond to the tutorial
workflow. :cite:`stelopt-neo-tutorial`

Run via CLI
-----------

.. code-block:: bash

   cd tests/fixtures/ncsx
   neo-jax ncsx_c09r00_free --boozmn boozmn_ncsx_c09r00_free.nc --verbose

Notes
-----

- The control file uses a fixed set of 10 surfaces.
- The Boozer file is large; GPU acceleration is recommended for full-resolution
  runs.
- The original tutorial output can be used as a reference for validating
  ``epstot`` across flux surfaces.
- For a JIT-based run and autodiff demo, see ``examples/ncsx_jit_run.py`` and
  ``examples/ncsx_autodiff_opt.py``.
