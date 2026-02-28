VMEC and Boozer Interfaces
==========================

NEO operates in Boozer coordinates, where the magnetic field can be written in
contravariant and covariant forms,

.. math::

   \mathbf{B} = \nabla \psi \times \nabla \theta_B + \iota \nabla \phi_B \times \nabla \psi
   = I(\psi) \nabla \phi_B + G(\psi) \nabla \theta_B + B_\psi \nabla \psi.

This representation is the basis for the BOOZ_XFORM transformation and the
``boozmn`` file format used by NEO. :cite:`booz-xform-jax,stelopt-neo-docs`

Required data for NEO
---------------------

NEO expects Fourier coefficients on each flux surface:

- ``rmnc``: cosine coefficients of cylindrical radius.
- ``zmns``: sine coefficients of vertical coordinate.
- ``lmns``: sine coefficients of the Boozer toroidal angle shift.
- ``bmnc``: cosine coefficients of magnetic field magnitude.
- ``ixm``, ``ixn``: poloidal and toroidal mode numbers.
- ``iota``: rotational transform profile.
- ``curr_pol`` and ``curr_tor``: Boozer currents :math:`I` and :math:`G`.
- ``nfp``: number of field periods.

Mapping from boozmn
-------------------

The standard ``boozmn`` netCDF file (from BOOZ_XFORM) provides arrays such as
``rmnc_b``, ``zmns_b``, ``bmnc_b``, ``pmns_b``, ``ixm_b``, ``ixn_b``, ``iota_b``,
``bvco_b``, and ``buco_b``. NEO maps these to its internal representation using

.. math::

   \lambda_{mn} = - \mathrm{pmns\_b}_{mn} \frac{n_{\mathrm{fp}}}{2\pi}.

The Boozer currents are mapped as ``curr_pol = bvco_b`` and
``curr_tor = buco_b``. The NEO_JAX reader mirrors this mapping.

End-to-end JAX pipeline
-----------------------

NEO_JAX is designed to consume the outputs of ``vmec_jax`` and
``booz_xform_jax`` directly, avoiding intermediate files and enabling
end-to-end differentiation. :cite:`vmec-jax,booz-xform-jax`

When using this pipeline, ensure the following:

- The Boozer transform uses the same field-period convention as NEO.
- Mode truncation matches ``max_m_mode`` and ``max_n_mode`` from the control file.
- The current profiles supplied to NEO are consistent with the Boozer convention.

NEO_JAX provides :func:`neo_jax.io.booz_xform_to_boozerdata` to convert arrays
from a booz_xform-style object into the ``BoozerData`` container used by the
solver, and :func:`neo_jax.run_neo` (or :func:`neo_jax.run_booz_xform`) to run
the solver directly on that object with a high-level configuration.

For pipeline workflows, see :func:`neo_jax.run_boozer_to_neo` and
:func:`neo_jax.run_vmec_boozer_neo` for convenience wrappers.

Example: vmec_jax → booz_xform_jax → neo_jax
--------------------------------------------

.. code-block:: python

   from neo_jax import NeoConfig, run_vmec_boozer_neo

   config = NeoConfig(surfaces=[0.25, 0.5, 0.75], theta_n=32, phi_n=32)
   results = run_vmec_boozer_neo(
       "path/to/input.vmec",
       vmec_kwargs=dict(max_iter=1, use_initial_guess=True, vmec_project=False),
       booz_kwargs=dict(mboz=8, nboz=8),
       neo_config=config,
   )

Notes:

- For accurate surface mapping, it is recommended to let the Boozer step
  compute all VMEC half-grid surfaces and use the NEO surface selection
  (``NeoConfig.surfaces``) to pick the subset.
- The pipeline avoids file I/O and returns JAX arrays, but full end-to-end JIT
  across vmec_jax is still in progress; see ``PLAN.md`` for roadmap details.
