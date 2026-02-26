Theory
======

NEO computes the effective helical ripple :math:`\epsilon_{\mathrm{eff}}^{3/2}`
used to characterize neoclassical transport in stellarators. The quantity is
defined in the ``1/\nu`` regime and described in the original NEO literature
and documentation. :cite:`nemov1999,stelopt-neo-docs`

Field-line integrals
--------------------

NEO follows magnetic field lines in Boozer coordinates and accumulates several
field-line integrals. The right-hand-side used in the field-line ODE integrates

.. math::

   y_1 &= \theta, \\
   y_2 &= \int d\phi\, B^{-2}, \\
   y_3 &= \int d\phi\, |\nabla \psi| B^{-2}, \\
   y_4 &= \int d\phi\, K_G B^{-3},

along with particle-dependent terms for trapped particles. These are computed
from the Boozer-field representation of :math:`B`, the geodesic curvature term
:math:`K_G`, and the parallel derivative of :math:`B`.

Pitch-angle sampling
--------------------

The trapped-particle contribution is evaluated by sampling a pitch parameter
:math:`\eta` between :math:`B_{\min}/B_0` and :math:`B_{\max}/B_0` on each flux
surface. For each :math:`\eta` the algorithm accumulates integrals of the form

.. math::

   I_f &= \int d\phi\, \sqrt{1 - B/B_0\eta}\, B^{-2}, \\
   H_f &= \int d\phi\, \sqrt{1 - B/B_0\eta}\,\left(\frac{4}{B/B_0} - \frac{1}{\eta}\right)
         \frac{K_G}{\sqrt{\eta}} B^{-2}.

Effective ripple
----------------

For each trapped-particle class, NEO computes

.. math::

   \epsilon_{\mathrm{eff}}^{3/2}(m) = C_\epsilon\,\frac{y_2}{y_3^2}\,\mathrm{BigInt}(m),

with

.. math::

   C_\epsilon = \frac{\pi R_0^2 \Delta\eta}{8\sqrt{2}},

and sums over classes to obtain the total
:math:`\epsilon_{\mathrm{eff}}^{3/2}`. Additional diagnostics (fraction of
trapped particles, ripple amplitudes, and :math:`y_4`) are reported when
``eout_swi = 2``.
