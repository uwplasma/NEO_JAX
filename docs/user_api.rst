User API Guide
==============

This page summarizes the high-level, user-friendly API introduced to make
NEO_JAX easy to use without manual control files.

Quick start
-----------

.. code-block:: python

   from neo_jax import NeoConfig, run_neo

   config = NeoConfig(surfaces=[19, 39, 59, 79], theta_n=64, phi_n=64)
   results = run_neo("boozmn.nc", config=config)

   print(results.epsilon_effective)
   print(results[0].epsilon_effective_by_class)

Accessing results
-----------------

``NeoResults`` behaves like a list of ``NeoSurfaceResult`` instances while also
supporting vector access by name:

.. code-block:: python

   eps_eff = results.epsilon_effective      # array over surfaces
   epspar = results["epsilon_effective_by_class"]  # stacked by surface

Aliases are supported for common names (``epstot`` → ``epsilon_effective``).

Configuration parameters
------------------------

The main solver parameters live in :class:`neo_jax.NeoConfig`:

- ``npart``: number of ``eta`` grid points (pitch parameter sampling).
- ``multra``: number of trapped-particle classes to resolve.
- ``nstep_per``: RK4 steps per field period.
- ``nstep_min`` / ``nstep_max``: minimum/maximum number of field periods to
  integrate before convergence checks.
- ``acc_req``: accuracy requirement for rational-surface handling.
- ``no_bins``: bins used for rational-surface coverage.

These map directly to the legacy NEO control file and can be overridden in
examples or custom workflows.

Radial coordinates
------------------

NEO_JAX reports multiple radial coordinates:

- ``s``: normalized toroidal flux (``0 ≤ s ≤ 1``). This is the default x-axis
  for plots and can be used to select surfaces.
- ``sqrt_s``: square root of ``s`` (often used as a proxy for minor radius).
- ``r_eff``: effective radius computed by integrating the NEO quantity
  ``dr/dψ`` over the flux grid. See :doc:`numerics` for details.

Running on Boozer objects
-------------------------

If you already have a Boozer object (for example, from ``booz_xform_jax``),
you can pass it directly to ``run_neo`` (or call ``run_booz_xform``):

.. code-block:: python

   from neo_jax import NeoConfig, run_neo

   config = NeoConfig(surfaces=[10, 20, 30])
   results = run_neo(booz_obj, config=config)

Pipeline helpers
----------------

For workflows that chain VMEC → Boozer → NEO, NEO_JAX provides two helpers:

- :func:`neo_jax.run_boozer_to_neo`: run NEO directly on a booz_xform output mapping.
- :func:`neo_jax.run_vmec_boozer_neo`: convenience wrapper for vmec_jax → booz_xform_jax → NEO.
- :func:`neo_jax.run_vmec_boozer_neo_jax`: JAX-native VMEC→Boozer adapter + JAX surface scan.

Example:

.. code-block:: python

   from neo_jax import NeoConfig, run_vmec_boozer_neo

   config = NeoConfig(surfaces=[0.25, 0.5, 0.75], theta_n=32, phi_n=32)
   results = run_vmec_boozer_neo(
       "path/to/input.vmec",
       vmec_kwargs=dict(max_iter=1, use_initial_guess=True, vmec_project=False),
       booz_kwargs=dict(mboz=8, nboz=8),
       neo_config=config,
   )

Plotting
--------

A helper is provided to plot epsilon effective vs radius:

.. code-block:: python

   from neo_jax import plot_epsilon_effective

   fig, ax = plot_epsilon_effective(results, x="s")
   fig.savefig("eps_eff.png", dpi=150)

Advanced workflows
------------------

For custom optimization loops (autodiff, JIT kernels, custom solvers), you can
use :func:`neo_jax.workflow.build_surface_problem` to construct the surface,
environment, and integration parameters in one step:

.. code-block:: python

   from neo_jax import NeoConfig, build_surface_problem
   from neo_jax.api import load_boozmn

   booz = load_boozmn("boozmn.nc")
   config = NeoConfig(surfaces=[0.35], theta_n=64, phi_n=64)
   problem = build_surface_problem(booz, config, surface=config.surfaces[0])

The returned ``SurfaceProblem`` contains the fields ``surface``, ``env``,
``params``, and ``Rmajor`` required by the low-level integrator.

Surface selection by ``s``
--------------------------

Surface selections may be specified by index or by normalized toroidal flux
``s`` (floats between 0 and 1). When floats are provided, NEO_JAX maps them to
the nearest available surface in the Boozer grid.

.. code-block:: python

   config = NeoConfig(surfaces=[0.2, 0.5, 0.8])
   results = run_neo("boozmn.nc", config=config)
