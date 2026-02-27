Quickstart
==========

File-based workflow (boozmn)
----------------------------

The simplest workflow is to read a ``boozmn`` file and run the solver on the
flux surfaces specified in a NEO control file.

.. code-block:: python

   from neo_jax.control import read_control
   from neo_jax.driver import run_neo_from_boozmn

   control = read_control("neo.in")
   results = run_neo_from_boozmn("boozmn.nc", control)

   for row in results:
       print(row["flux_index"], row["epstot"], row["reff"])

To emulate ``xneo``, you can use the CLI:

.. code-block:: bash

   neo-jax ORBITS --boozmn boozmn_ORBITS.nc --verbose

Examples
--------

Two executable examples are included in ``examples/``:

- ``examples/ncsx_jit_run.py``: run a JIT-compiled NCSX solve on a single surface.
- ``examples/ncsx_autodiff_opt.py``: minimize a scalar loss using autodiff through
  ``flint_bo_jax``.

JAX-native workflow (VMEC + Boozer + NEO)
-----------------------------------------

NEO_JAX is designed to consume JAX-native VMEC and Boozer transform outputs,
allowing end-to-end differentiation when paired with ``vmec_jax`` and
``booz_xform_jax``. :cite:`vmec-jax,booz-xform-jax`

A typical flow is:

- Use ``vmec_jax`` to compute an equilibrium state from a VMEC input file.
- Use ``booz_xform_jax`` to generate Boozer Fourier coefficients and currents.
- Pass those arrays directly into :mod:`neo_jax.driver` without writing
  ``wout`` or ``boozmn`` files.

See :doc:`vmec_boozer` for the required data interface and mapping details.
