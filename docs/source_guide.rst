Source Guide
============

This page is a code-oriented map of NEO_JAX for readers who want to connect the
theory and numerics pages to the implementation.

Module structure
----------------

The repository is organized into a small number of solver-facing modules:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Module
     - Responsibility
   * - ``neo_jax.api``
     - High-level public API such as :func:`neo_jax.run_neo` and convenience
       wrappers for ``boozmn`` files and in-memory Boozer objects.
   * - ``neo_jax.config``
     - User-facing configuration model.
   * - ``neo_jax.io``
     - ``boozmn`` loading and conversion from booz_xform-style objects.
   * - ``neo_jax.fourier``
     - Fourier reconstruction and derived geometric quantities.
   * - ``neo_jax.surface``
     - Surface initialization, spline construction, and
       :math:`B_{\min}`/:math:`B_{\max}` refinement.
   * - ``neo_jax.geometry``
     - Spline evaluation and Newton-based extremum refinement.
   * - ``neo_jax.integrate``
     - Field-line RHS, RK4 stepping, trapped-particle bookkeeping, and the JAX
       scan backend.
   * - ``neo_jax.driver``
     - Surface loop orchestration, scaling, diagnostics, and result assembly.
   * - ``neo_jax.pipeline``
     - VMEC→Boozer→NEO helper workflows.

Geometry loading
----------------

The ``boozmn`` reader is where the external geometry enters the solver:

.. literalinclude:: ../neo_jax/io.py
   :language: python
   :pyobject: read_boozmn

This function:

- resolves the requested file path
- loads the Boozer Fourier coefficients and current profiles
- maps the ``boozmn`` convention to the internal :class:`neo_jax.BoozerData`
  container
- computes the normalized toroidal-flux coordinate ``s`` used by the public API

Surface initialization
----------------------

For each selected surface, NEO_JAX reconstructs the geometry, derives the
metric quantities, builds the spline representation, and refines
:math:`B_{\min}` and :math:`B_{\max}`:

.. literalinclude:: ../neo_jax/surface.py
   :language: python
   :pyobject: init_surface

Field-line RHS
--------------

The core continuous model enters through :func:`neo_jax.integrate.rhs_bo1`:

.. literalinclude:: ../neo_jax/integrate.py
   :language: python
   :pyobject: rhs_bo1

This is where the state vector
:math:`(\theta, y_2, y_3, y_4, I_j, H_j)` is advanced and where the trapped
particle masks are updated.

JAX scan backend
----------------

The compiled backend lives in :func:`neo_jax.integrate.flint_bo_jax`:

.. literalinclude:: ../neo_jax/integrate.py
   :language: python
   :pyobject: flint_bo_jax

This backend keeps the dominant loops on device, which is the basis for JIT
reuse and batched surface evaluation.

Public configuration model
--------------------------

The public configuration surface is intentionally compact:

.. literalinclude:: ../neo_jax/config.py
   :language: python
   :pyobject: NeoConfig

For user-level guidance on these fields, see :doc:`configuration`.

Pipeline entrypoints
--------------------

The reusable VMEC→Boozer→NEO path is built in ``neo_jax.pipeline``:

.. literalinclude:: ../neo_jax/pipeline.py
   :language: python
   :pyobject: build_vmec_boozer_neo_jax

This callable is the preferred entrypoint for repeated JAX-native studies where
the same static geometry setup is reused many times.

Reference crosswalk
-------------------

For readers comparing NEO_JAX to the established STELLOPT implementation,
the existing routine-level crosswalk remains available in :doc:`fortran_map`.
