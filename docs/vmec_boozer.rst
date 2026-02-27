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
solver, and :func:`neo_jax.run_booz_xform` to run the solver directly on that
object with a high-level configuration.
