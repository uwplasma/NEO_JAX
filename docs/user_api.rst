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

   fig, ax = plot_epsilon_effective(results)
   fig.savefig("eps_eff.png", dpi=150)
