Overview
========

NEO_JAX computes effective helical ripple and related trapped-particle
diagnostics for stellarator equilibria represented in Boozer coordinates. The
main target quantity is
:math:`\epsilon_{\mathrm{eff}}^{3/2}`, the transport measure that enters the
:math:`1/\nu` neoclassical regime. :cite:`nemov1999,stelopt-neo-docs`

Scientific scope
----------------

Given a Boozer-coordinate representation of an equilibrium,
NEO_JAX reconstructs the geometry on each selected flux surface, follows field
lines, evaluates trapped-particle line integrals, and assembles:

- total :math:`\epsilon_{\mathrm{eff}}^{3/2}`
- trapped-class contributions ``epspar``
- trapped fractions ``ctrone`` and ``ctrtot``
- ripple amplitudes ``bareph`` and ``barept``
- radial coordinates ``s``, ``sqrt_s``, and ``r_eff``

The code is intended for:

- standalone post-processing of ``boozmn`` files
- direct operation on in-memory Boozer arrays
- automated parameter scans and optimization workflows
- JAX-based studies that benefit from JIT compilation or differentiation

Supported workflows
-------------------

NEO_JAX supports three main user workflows:

1. **File-based solver**:
   read ``boozmn`` files and use the Python API or CLI.
2. **In-memory Boozer workflow**:
   pass arrays from ``booz_xform_jax`` directly into :func:`neo_jax.run_neo`.
3. **End-to-end VMEC→Boozer→NEO workflow**:
   compose ``vmec_jax``, ``booz_xform_jax``, and NEO_JAX without intermediate
   files. :cite:`vmec-jax,booz-xform-jax`

NEO_JAX also preserves support for standard ``neo_in.*`` and ``neo_param.*``
input decks when a file-oriented workflow is preferred.

Design principles
-----------------

The current implementation is organized around a few engineering principles:

- **transparent numerics**:
  the continuous model and the discrete algorithm are documented and exposed
  through a readable Python/JAX implementation
- **explicit geometry model**:
  Boozer-coordinate inputs, derived metric quantities, and solver outputs are
  named directly in the API
- **JAX where it helps**:
  device-resident loops, reusable compiled kernels, and autodiff support are
  provided without forcing every workflow into a single execution mode
- **reference-backed validation**:
  curated fixture comparisons and regression tests are used to validate the
  implementation on representative cases

Where to read next
------------------

- :doc:`theory` for the governing transport model and the integral quantities
- :doc:`numerics` for the discrete algorithm
- :doc:`configuration` for solver knobs and runtime controls
- :doc:`vmec_boozer` for geometry inputs and pipeline composition
- :doc:`testing` and :doc:`validation` for the verification story
