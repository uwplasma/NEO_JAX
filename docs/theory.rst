Theory
======

NEO_JAX evaluates the effective ripple measure
:math:`\epsilon_{\mathrm{eff}}^{3/2}` used in stellarator neoclassical
transport theory in the :math:`1/\nu` regime. The governing model follows the
analytic treatment of trapped-particle transport in Boozer coordinates. :cite:`nemov1999`

Boozer-coordinate representation
--------------------------------

The solver assumes that each flux surface is provided in Boozer coordinates
:math:`(\psi,\theta_B,\phi_B)` with magnetic field magnitude represented as a
Fourier series:

.. math::

   B(\theta_B,\phi_B;\psi)
   = \sum_{m,n} B_{mn}(\psi)\cos(m\theta_B - n\phi_B).

The magnetic field can be written in the usual covariant/contravariant Boozer
forms, which is the representation produced by Boozer transforms of VMEC
equilibria:

.. math::

   \mathbf{B}
   = \nabla\psi \times \nabla\theta_B
   + \iota(\psi)\,\nabla\phi_B \times \nabla\psi
   = G(\psi)\nabla\theta_B + I(\psi)\nabla\phi_B + B_\psi \nabla\psi.

Here :math:`\iota` is the rotational transform, while :math:`I` and :math:`G`
are the Boozer current functions supplied by the geometry input. In NEO_JAX
they appear as ``curr_pol`` and ``curr_tor``.

Continuous state integrated along a field line
----------------------------------------------

The line-following part of the algorithm advances a state vector
:math:`y(\phi_B)` containing a geometric angle plus several accumulated
integrals:

.. math::

   y = \left(
      \theta,\,
      Y_2,\,
      Y_3,\,
      Y_4,\,
      I_1,\ldots,I_{N_\eta},\,
      H_1,\ldots,H_{N_\eta}
   \right).

The first four components satisfy

.. math::

   \frac{d\theta}{d\phi_B} &= \iota, \\
   \frac{dY_2}{d\phi_B} &= B^{-2}, \\
   \frac{dY_3}{d\phi_B} &= |\nabla\psi|\,B^{-2}, \\
   \frac{dY_4}{d\phi_B} &= K_G\,B^{-3},

where :math:`K_G` is the geodesic-curvature term used in the Nemov effective
ripple formula.

Pitch-parameter sampling
------------------------

Trapped-particle contributions are sampled on a pitch grid
:math:`\eta_j \in [B_{\min}/B_0,\, B_{\max}/B_0]`, where
:math:`B_0 = B_{\max}` on the surface in the default scaling. For each pitch
sample, NEO_JAX evaluates masked line integrals of the form

.. math::

   \chi_j(\phi_B) &=
   \begin{cases}
   1, & 1 - B/(B_0\eta_j) > 0, \\
   0, & \text{otherwise},
   \end{cases} \\
   \frac{dI_j}{d\phi_B} &= \chi_j
      \sqrt{1 - \frac{B}{B_0\eta_j}}\,B^{-2}, \\
   \frac{dH_j}{d\phi_B} &= \chi_j
      \sqrt{1 - \frac{B}{B_0\eta_j}}
      \left(\frac{4}{B/B_0} - \frac{1}{\eta_j}\right)
      \frac{K_G}{\sqrt{\eta_j}}\,B^{-2}.

The mask :math:`\chi_j` identifies the trapped region for that pitch sample.
The code tracks sign changes in the parallel derivative of :math:`B` to assign
samples to trapped-particle classes.

Effective ripple assembly
-------------------------

The trapped-particle accumulation produces a class-dependent quantity
``BigInt(m)`` in the code. The reported effective-ripple contribution for class
:math:`m` is

.. math::

   \epsilon_{\mathrm{eff},m}^{3/2}
   = C_\epsilon \frac{Y_2}{Y_3^2}\,\mathrm{BigInt}(m),

with the quadrature prefactor

.. math::

   C_\epsilon = \frac{\pi R_0^2 \Delta\eta}{8\sqrt{2}},

where :math:`R_0` is the reference major radius and :math:`\Delta\eta` is the
uniform pitch-grid spacing.

The total reported quantity is

.. math::

   \epsilon_{\mathrm{eff}}^{3/2}
   = \sum_{m=1}^{m_{\max}}
   \epsilon_{\mathrm{eff},m}^{3/2}.

Additional reported diagnostics
-------------------------------

In addition to :math:`\epsilon_{\mathrm{eff}}^{3/2}`, NEO_JAX reports:

- ``epspar``:
  class-wise contributions to effective ripple
- ``ctrone``:
  fraction of singly trapped particles
- ``ctrtot``:
  fraction of all trapped particles encountered in the sampled population
- ``bareph`` and ``barept``:
  ripple-amplitude proxies derived from the trapped fractions
- ``yps``:
  the accumulated :math:`Y_4`-related transport diagnostic

Reference scaling
-----------------

The final reported values are scaled by the selected reference field and radius:

.. math::

   \epsilon_{\mathrm{eff,reported}}^{3/2}
   =
   \epsilon_{\mathrm{eff,raw}}^{3/2}
   \left(\frac{B_\mathrm{ref}}{B_{\max}}\right)^2
   \left(\frac{R_\mathrm{ref}}{R_0}\right)^2.

This is controlled by ``ref_swi`` and is described in :doc:`configuration`.
