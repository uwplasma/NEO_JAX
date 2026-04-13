Applications and Examples
=========================

NEO_JAX is designed both as a production solver for Boozer-coordinate
transport analysis and as a building block for optimization, design-space
exploration, and differentiable workflows.

Example scripts
---------------

The repository ships executable examples in ``examples/``:

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Script
     - Purpose
   * - ``ncsx_epsilon_effective_plot.py``
     - Computes :math:`\epsilon_{\mathrm{eff}}^{3/2}` on an NCSX fixture and
       plots the radial profile.
   * - ``ncsx_autodiff_Rmajor_optimization.py``
     - Demonstrates forward-mode differentiation with respect to a simple
       geometric scalar.
   * - ``epsilon_effective_scale_optimization.py``
     - Uses autodiff on a toy magnetic-field scaling problem.
   * - ``qh_epsilon_effective_aspect_optimization.py``
     - Combines effective ripple with aspect-ratio objectives in a warm-start
       optimization setting.
   * - ``vmec_boozer_neo_pipeline.py``
     - Runs the in-memory VMEC→Boozer→NEO workflow without intermediate files.

Typical research uses
---------------------

NEO_JAX is well suited to:

- radial scans of :math:`\epsilon_{\mathrm{eff}}^{3/2}` on a fixed equilibrium
- design comparisons across Boozer or VMEC equilibria
- sensitivity analysis of ripple to geometry or current-profile changes
- gradient-based optimization where a differentiable approximation is useful
- automated studies in which the Python API is more convenient than text-file
  orchestration
- solver benchmarking across CPU, GPU, JIT, and Fourier-evaluation modes

Example: radial survey from a Boozer file
-----------------------------------------

.. code-block:: python

   from neo_jax import NeoConfig, run_neo

   config = NeoConfig(
       surfaces=[0.1, 0.2, 0.4, 0.6, 0.8],
       theta_n=96,
       phi_n=96,
       npart=64,
       multra=2,
   )
   results = run_neo("boozmn.nc", config=config)

   for s, eps in zip(results.s, results.epsilon_effective):
       print(f"s={s:.3f}  eps_eff^(3/2)={eps:.6e}")

Example: in-memory pipeline
---------------------------

.. code-block:: python

   from neo_jax import NeoConfig, run_vmec_boozer_neo

   config = NeoConfig(surfaces=[0.3, 0.5, 0.7], theta_n=64, phi_n=64)
   results = run_vmec_boozer_neo(
       "input.vmec",
       vmec_kwargs=dict(max_iter=1, use_initial_guess=True, vmec_project=False),
       booz_kwargs=dict(mboz=12, nboz=12),
       neo_config=config,
   )

Optimization and autodiff
-------------------------

The main autodiff use cases in the repository are deliberately simple and
transparent:

- changing a scalar geometry parameter and differentiating the resulting
  :math:`\epsilon_{\mathrm{eff}}^{3/2}`
- reusing a prebuilt JAX solver inside an outer optimization loop
- studying tradeoffs between ripple and secondary geometric objectives

For repeated solves, prefer :func:`neo_jax.build_vmec_boozer_neo_jax` so the
static geometry setup is compiled once and reused:

.. code-block:: python

   from neo_jax import NeoConfig, build_vmec_boozer_neo_jax

   solve = build_vmec_boozer_neo_jax(
       run,
       booz_kwargs=dict(mboz=10, nboz=10),
       neo_config=NeoConfig(surfaces=[0.5]),
       jit=True,
   )
   outputs = solve(run.state)

Comparison workflows
--------------------

NEO_JAX also supports direct comparison against existing output decks and
reference cases when that is scientifically useful. Those comparisons are
documented in :doc:`validation` and :doc:`testing`; they are intended as
evidence for correctness, not as the primary framing of the code.
