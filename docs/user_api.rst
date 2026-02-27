User API Guide
==============

This page summarizes the high-level, user-friendly API introduced to make
NEO_JAX easy to use without manual control files.

Quick start
-----------

.. code-block:: python

   from neo_jax import NeoConfig, run_boozmn

   config = NeoConfig(surfaces=[19, 39, 59, 79], theta_n=64, phi_n=64)
   results = run_boozmn("boozmn.nc", config=config)

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

If you already have a Boozer object (for example, from ``booz_xform_jax``), use
``run_booz_xform``:

.. code-block:: python

   from neo_jax import NeoConfig, run_booz_xform

   config = NeoConfig(surfaces=[10, 20, 30])
   results = run_booz_xform(booz_obj, config=config)

Plotting
--------

A helper is provided to plot epsilon effective vs radius:

.. code-block:: python

   from neo_jax import plot_epsilon_effective

   fig, ax = plot_epsilon_effective(results, x="s")
   fig.savefig("eps_eff.png", dpi=150)

Surface selection by ``s``
--------------------------

Surface selections may be specified by index or by normalized toroidal flux
``s`` (floats between 0 and 1). When floats are provided, NEO_JAX maps them to
the nearest available surface in the Boozer grid.

.. code-block:: python

   config = NeoConfig(surfaces=[0.2, 0.5, 0.8])
   results = run_boozmn("boozmn.nc", config=config)
