User API Guide
==============

This page covers the main Python entrypoints for NEO_JAX and the data objects
they exchange.

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

   eps_eff = results.epsilon_effective
   epspar = results["epsilon_effective_by_class"]
   r_eff = results.r_eff
   diag0 = results[0].diagnostics

Aliases are supported for common names (``epstot`` → ``epsilon_effective``).

The most important result fields are:

- ``epsilon_effective``:
  total :math:`\epsilon_{\mathrm{eff}}^{3/2}` over surfaces
- ``epsilon_effective_by_class``:
  trapped-class contributions
- ``s``, ``sqrt_s``, ``r_eff``:
  radial coordinate options for plotting and analysis
- ``iota``, ``b_ref``, ``r_ref``:
  surface metadata and scaling quantities
- ``diagnostics``:
  per-surface dictionary of additional solver metadata

Main entrypoints
----------------

The public API revolves around a few convenience functions:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Function
     - Purpose
   * - :func:`neo_jax.run_neo`
     - General entrypoint accepting a ``boozmn`` path, ``BoozerData``, or
       booz_xform-style object.
   * - :func:`neo_jax.run_boozmn`
     - Convenience wrapper for file-based ``boozmn`` input.
   * - :func:`neo_jax.run_boozer`
     - Run directly on a :class:`neo_jax.BoozerData` instance.
   * - :func:`neo_jax.run_booz_xform`
     - Run on a booz_xform-style mapping or object.
   * - :func:`neo_jax.build_surface_problem`
     - Construct a single-surface bundle for advanced custom workflows.

Configuration parameters
------------------------

Solver settings live in :class:`neo_jax.NeoConfig`. The most important fields
for scientific use are:

- resolution: ``theta_n``, ``phi_n``
- pitch-grid and class controls: ``npart``, ``multra``
- field-line integration controls:
  ``nstep_per``, ``nstep_min``, ``nstep_max``, ``acc_req``, ``no_bins``
- radial selection: ``surfaces``
- low-``|iota|`` behavior:
  ``max_rational_field_periods`` and ``rational_surface_policy``

See :doc:`configuration` for a full control summary.

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
- :func:`neo_jax.build_vmec_boozer_neo_jax`: build a reusable JAX-native pipeline callable.

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

When ``jax_surface_scan=True`` (or when using :func:`neo_jax.run_vmec_boozer_neo_jax`),
the return type is a JAX-friendly :class:`neo_jax.data_models.NeoOutputs` with
arrays in JAX device memory. This is useful for autodiff pipelines; you can
convert to NumPy as needed. For convenience, use
:func:`neo_jax.neo_outputs_to_results` to obtain the standard
:class:`neo_jax.results.NeoResults` container:

.. code-block:: python

   from neo_jax import neo_outputs_to_results

   results = neo_outputs_to_results(outputs)

Reusable JAX pipeline
---------------------

For optimization loops or repeated solves, build the pipeline once and reuse
the returned callable (optionally JIT-compiled):

.. code-block:: python

   from neo_jax import NeoConfig, build_vmec_boozer_neo_jax

   solver = build_vmec_boozer_neo_jax(
       run,
       booz_kwargs=dict(mboz=8, nboz=8),
       neo_config=NeoConfig(surfaces=[0.5]),
       jit=True,
   )
   outputs = solver(run.state)

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

Choosing between Python and JAX backends
----------------------------------------

The API exposes both execution modes:

- ``use_jax=True`` selects the compiled JAX implementation
- ``use_jax=False`` uses the Python-loop implementation
- ``jax_surface_scan=True`` batches surfaces in the JAX path when supported

This makes it straightforward to move between:

- easy debugging in Python mode
- reproducible one-off solves in standard JAX mode
- repeated compiled solves in batched JAX mode
