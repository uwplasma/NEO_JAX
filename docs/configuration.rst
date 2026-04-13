Configuration and Runtime Controls
==================================

NEO_JAX exposes the solver through a compact high-level configuration object,
an ``xneo``-style CLI, and a small set of environment variables for runtime
selection. This page collects those controls in one place.

High-Level Python Configuration
-------------------------------

The main user entrypoint is :class:`neo_jax.NeoConfig`.

.. list-table::
   :header-rows: 1
   :widths: 20 45 35

   * - Field
     - Meaning
     - Typical usage
   * - ``surfaces``
     - Flux surfaces to analyze, either as 1-based indices or normalized
       toroidal-flux values ``s`` in ``[0, 1]``.
     - Use ``[0.25, 0.5, 0.75]`` for user-facing scripts; use explicit indices
       when reproducing a fixed input deck.
   * - ``theta_n``, ``phi_n``
     - Angular grid resolution used for Fourier reconstruction and spline
       construction on each surface.
     - ``64``–``200`` for production studies, depending on geometry complexity.
   * - ``max_m_mode``, ``max_n_mode``
     - Optional Fourier truncation applied to the Boozer spectrum before field
       reconstruction.
     - Leave at ``0`` to use all available modes.
   * - ``npart``
     - Number of pitch-parameter samples :math:`\eta` between
       :math:`B_{\min}/B_0` and :math:`B_{\max}/B_0`.
     - Increase for smoother trapped-particle quadrature.
   * - ``multra``
     - Maximum trapped-particle class index accumulated in
       :math:`\epsilon_{\mathrm{eff}}^{3/2}`.
     - ``1`` for singly trapped only, ``2`` to include doubly trapped, and so on.
   * - ``acc_req``
     - Accuracy target for rational-surface coverage and convergence checks.
     - Smaller values increase runtime, especially when ``|iota|`` is small.
   * - ``no_bins``
     - Number of poloidal bins used when deciding whether a rational surface has
       been covered adequately.
     - ``50``–``100`` is typical.
   * - ``nstep_per``
     - RK4 steps per field period.
     - Increase for more difficult surfaces or stronger mode content.
   * - ``nstep_min``, ``nstep_max``
     - Minimum and maximum number of field periods integrated before stopping.
     - ``nstep_max`` is the hard cap for expensive surfaces.
   * - ``calc_nstep_max``
     - Current-path control used when the parallel-current solve is active.
     - Leave at the default unless reproducing a known control deck.
   * - ``max_rational_field_periods``
     - Preflight ceiling on the estimated rational-surface workload.
     - Keep enabled by default; set to ``0`` only for the full exact long run.
   * - ``rational_surface_policy``
     - Policy when the preflight estimate exceeds the ceiling.
     - ``"error"`` for fail-fast, ``"approximate"`` for controlled fallback.
   * - ``ref_swi``
     - Reference scaling for the reported :math:`B_\mathrm{ref}`.
     - ``2`` uses the flux-surface maximum and is the default.
   * - ``write_progress``, ``write_diagnostic``
     - Enable progress prints and optional diagnostic file generation.
     - Most Python API workflows leave these off.

Common Python patterns:

.. code-block:: python

   from neo_jax import NeoConfig, run_neo

   config = NeoConfig(
       surfaces=[0.2, 0.5, 0.8],
       theta_n=96,
       phi_n=96,
       npart=64,
       multra=2,
       acc_req=0.01,
   )
   results = run_neo("boozmn.nc", config=config)

Runtime selection flags
-----------------------

The main runtime arguments are carried by :func:`neo_jax.run_neo`.

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Argument
     - Meaning
     - Notes
   * - ``use_jax``
     - Select the JAX backend instead of the Python-loop backend.
     - Default is ``True``.
   * - ``progress``
     - Print high-level status and per-surface preflight information.
     - Useful for long runs and debugging.
   * - ``jax_surface_scan``
     - Evaluate multiple surfaces with the batched JAX scan path.
     - Best for JIT workflows and repeated calls.
   * - ``max_m_mode``, ``max_n_mode``
     - One-shot mode truncation overrides.
     - Usually leave these in ``NeoConfig`` instead.

Environment variables
---------------------

The runtime also recognizes several environment variables:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Variable
     - Effect
   * - ``NEO_JAX_ENABLE_X64``
     - Enable 64-bit floating point in JAX by default for production accuracy.
   * - ``JAX_ENABLE_X64``
     - Global JAX precision override.
   * - ``NEO_JAX_DISABLE_JIT``
     - Force eager execution for debugging and tracing.
   * - ``NEO_JAX_FOURIER_MODE``
     - Choose between ``vectorized`` and ``streamed`` Fourier reconstruction.
   * - ``NEO_JAX_MAX_RATIONAL_FIELD_PERIODS``
     - Set the preflight rational-work ceiling globally.
   * - ``NEO_JAX_RATIONAL_SURFACE_POLICY``
     - Choose ``error`` or ``approximate`` globally.
   * - ``NEO_JAX_WRITE_IPMAX_DEBUG``
     - Emit extra trapped-particle convergence diagnostics.
   * - ``NEO_JAX_RUN_GPU``
     - Enable optional GPU smoke tests in the regression suite.
   * - ``NEO_JAX_RUN_SLOW``
     - Enable slow reference-comparison cases.
   * - ``NEO_REFERENCE_BIN``
     - Override the reference executable used in CLI comparison tests.

Near-zero-``|iota|`` controls
-----------------------------

The most important runtime knobs for difficult surfaces are:

.. math::

   n_{\mathrm{fp,rat}} \sim \left\lceil \frac{1}{\mathrm{acc\_req}\,|\iota|} \right\rceil.

When ``|iota|`` is very small, the rational-surface correction can request a
very large number of field periods. NEO_JAX therefore separates three modes of
operation:

- guarded exact workflow:
  keep ``rational_surface_policy="error"`` and a finite
  ``max_rational_field_periods``
- controlled fallback:
  keep the ceiling but set ``rational_surface_policy="approximate"``
- full exact long run:
  set ``max_rational_field_periods=0``

The approximate mode keeps the base integration result and skips only the
expensive rational-surface correction when the preflight estimate is too large.
Whenever this happens, the terminal output and diagnostics say so explicitly.

CLI options
-----------

The command-line interface retains the standard ``xneo`` positional usage but
also adds explicit flags:

- ``--control``: select a specific control file
- ``--boozmn``: override the resolved Boozer input path
- ``--output``: override the output filename
- ``--jax`` / ``--no-jax``: choose the backend
- ``--verbose`` / ``--quiet``: control progress output

For CLI-specific details and output-file conventions, see :doc:`cli`.

Outputs and diagnostics
-----------------------

The public Python API returns :class:`neo_jax.NeoResults`, a sequence-like
container with vector accessors such as:

- ``epsilon_effective``
- ``epsilon_effective_by_class``
- ``s``, ``sqrt_s``, ``r_eff``
- ``iota``, ``b_ref``, ``r_ref``
- ``ctrone``, ``ctrtot``, ``bareph``, ``barept``, ``yps``

Each surface also carries a ``diagnostics`` mapping. This is where NEO_JAX
stores extra information such as the rational-work estimate, the selected
reference policy, and whether the approximate low-``|iota|`` path was used.
