Quickstart
==========

This page shows the three quickest ways to start using NEO_JAX: from a
``boozmn`` file, from the command line, and from an in-memory geometry object.

Install from PyPI with:

.. code-block:: bash

   pip install neo-jax

Python API from ``boozmn``
--------------------------

The simplest workflow is to read a ``boozmn`` file and run the solver with a
high-level config.

.. code-block:: python

   from neo_jax import NeoConfig, run_neo

   # Surfaces may be specified by index or by s in [0, 1].
   config = NeoConfig(surfaces=[0.15, 0.35, 0.6, 0.85], theta_n=64, phi_n=64)
   results = run_neo("boozmn.nc", config=config)

   # Access by name (aliases supported)
   print(results.epsilon_effective)
   print(results["epsilon_effective_by_class"].shape)

To emulate ``xneo``, you can use the CLI:

.. code-block:: bash

   xneo example_case
   xneo_jax example_case
   python -m neo_jax example_case

This interface reads the standard control-file layouts and output filenames for
file-based NEO workflows. See :doc:`cli` for the full command-line contract.

In-memory geometry workflow
---------------------------

If you already have a Boozer object or booz_xform-style mapping, pass it
directly:

.. code-block:: python

   from neo_jax import NeoConfig, run_neo

   config = NeoConfig(surfaces=[1, 2, 3], theta_n=64, phi_n=64)
   results = run_neo(booz_obj, config=config)

Examples
--------

Executable examples are included in ``examples/``:

- ``examples/ncsx_epsilon_effective_plot.py``: compute and plot epsilon effective vs ``s``.
- ``examples/ncsx_autodiff_Rmajor_optimization.py``: autodiff demo that adjusts
  ``Rmajor`` to reduce epsilon effective (toy example).
- ``examples/epsilon_effective_scale_optimization.py``: toy optimization that
  scales ``|B|`` to reduce epsilon effective (autodiff demo).
- ``examples/qh_epsilon_effective_aspect_optimization.py``: QH warm-start
  optimization (epsilon effective + aspect ratio).

JAX-native workflow (VMEC + Boozer + NEO)
-----------------------------------------

NEO_JAX is designed to consume JAX-native VMEC and Boozer transform outputs,
allowing end-to-end differentiation when paired with ``vmec_jax`` and
``booz_xform_jax``. :cite:`vmec-jax,booz-xform-jax`

A typical flow is:

- Use ``vmec_jax`` to compute an equilibrium state from a VMEC input file.
- Use ``booz_xform_jax.jax_api`` to generate Boozer Fourier coefficients and currents.
- Pass those arrays directly into :func:`neo_jax.run_neo` (or
  :func:`neo_jax.run_boozer_to_neo`) without writing ``wout`` or ``boozmn`` files.

For a convenience wrapper that runs vmec_jax → booz_xform_jax → NEO in one call:

.. code-block:: python

   from neo_jax import NeoConfig, run_vmec_boozer_neo

   config = NeoConfig(surfaces=[0.25, 0.5, 0.75], theta_n=32, phi_n=32)
   results = run_vmec_boozer_neo(
       "path/to/input.vmec",
       vmec_kwargs=dict(max_iter=1, use_initial_guess=True, vmec_project=False),
       booz_kwargs=dict(mboz=8, nboz=8),
       neo_config=config,
   )

For repeated solves or optimization loops, build a reusable JAX pipeline:

.. code-block:: python

   from neo_jax import build_vmec_boozer_neo_jax, NeoConfig

   solver = build_vmec_boozer_neo_jax(
       run,
       booz_kwargs=dict(mboz=8, nboz=8),
       neo_config=NeoConfig(surfaces=[0.5]),
       jit=True,
   )
   outputs = solver(run.state)

For a JAX-native VMEC→Boozer adapter plus JAX surface scan (no NumPy in VMEC→Boozer),
use :func:`neo_jax.run_vmec_boozer_neo_jax` on a `vmec_jax.FixedBoundaryRun`.

See :doc:`vmec_boozer` for the required data interface and mapping details.

Low-``|iota|`` note
-------------------

If the requested geometry contains surfaces with very small ``|iota|``, the
exact rational-surface correction can be extremely expensive. NEO_JAX therefore
supports:

- fail-fast guarded execution (default)
- a controlled fallback with ``rational_surface_policy="approximate"``
- the full exact long run with ``max_rational_field_periods=0``

See :doc:`configuration` and :doc:`numerics` for the details.
