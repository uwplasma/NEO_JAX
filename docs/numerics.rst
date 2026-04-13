Numerics
========

Grid and Fourier reconstruction
-------------------------------

Each selected surface is reconstructed on a tensor-product grid

.. math::

   \theta_i = \frac{2\pi i}{N_\theta - 1},
   \qquad
   \phi_j = \frac{2\pi j}{n_{\mathrm{fp}}(N_\phi - 1)},

for ``theta_n = N_\theta`` and ``phi_n = N_\phi``. One toroidal field period
is covered in :math:`\phi`.

The primary Fourier reconstruction is

.. math::

   R(\theta,\phi) &= \sum_{m,n} R_{mn}\cos(m\theta - n\phi), \\
   Z(\theta,\phi) &= \sum_{m,n} Z_{mn}\sin(m\theta - n\phi), \\
   \lambda(\theta,\phi) &= \sum_{m,n} \lambda_{mn}\sin(m\theta - n\phi), \\
   B(\theta,\phi) &= \sum_{m,n} B_{mn}\cos(m\theta - n\phi).

NEO_JAX evaluates these sums in two ways:

- ``vectorized`` mode:
  allocates :math:`\theta \times \phi \times \text{mode}` temporaries and is
  generally fastest
- ``streamed`` mode:
  loops over modes with ``jax.lax.fori_loop`` and uses less memory

The code also evaluates the derivatives required for the metric:

.. math::

   R_\theta,\ R_\phi,\ Z_\theta,\ Z_\phi,\ \lambda_\theta,\ \lambda_\phi,\
   B_\theta,\ B_\phi.

Derived metric quantities
-------------------------

Once the basic Fourier fields are reconstructed, NEO_JAX forms the covariant
metric coefficients used by the solver:

.. math::

   g_{\theta\theta} &= R_\theta^2 + Z_\theta^2 + R^2 \lambda_\theta^2, \\
   g_{\phi\phi} &= R_\phi^2 + Z_\phi^2 + R^2 \lambda_\phi^2, \\
   g_{\theta\phi} &= R_\theta R_\phi + Z_\theta Z_\phi + R^2 \lambda_\theta \lambda_\phi.

The code stores these as ``gtbtb``, ``gpbpb``, and ``gtbpb``. It then forms

.. math::

   \mathrm{fac} &= I + \iota G, \\
   \mathrm{isqrg} &= \frac{B^2}{\mathrm{fac}}, \\
   \sqrt{g^{11}} &= \sqrt{g_{\theta\theta}g_{\phi\phi} - g_{\theta\phi}^2}\,\mathrm{isqrg}, \\
   K_G &= \frac{G\,B_\phi - I\,B_\theta}{I + \iota G}, \\
   \partial_\parallel B &= B_\phi + \iota B_\theta.

In the implementation, these appear as ``sqrg11``, ``kg``, and ``pard``.

Spline representation
---------------------

After Fourier reconstruction, the fields are converted into periodic 2D cubic
splines. This is the main interpolation layer used during field-line
integration:

- ``b_spl`` for :math:`B`
- ``g_spl`` for :math:`\sqrt{g^{11}}`
- ``k_spl`` for :math:`K_G`
- ``p_spl`` for :math:`\partial_\parallel B`
- ``q_spl`` for the current-related path when needed

This split is important numerically:

- the expensive Fourier evaluation is done once per surface
- the ODE right-hand side then uses local spline evaluation
- the resulting integration cost scales primarily with the line-following work,
  not with repeated mode summation

Finding :math:`B_{\min}` and :math:`B_{\max}`
---------------------------------------------

The solver starts from the grid extrema of :math:`B` and refines them with a
Newton solve on the spline representation. Denoting
:math:`f = \partial B/\partial\theta` and
:math:`g = \partial B/\partial\phi`, the root finder solves

.. math::

   f(\theta,\phi) = 0,
   \qquad
   g(\theta,\phi) = 0

using the local Jacobian of first and second spline derivatives. The resulting
``theta_bmin``, ``phi_bmin``, ``theta_bmax``, and ``phi_bmax`` are then used to
evaluate the refined extrema.

Field-line integration
----------------------

The ODE system is advanced with classical fourth-order Runge-Kutta. For a step
size :math:`h`, the code computes

.. math::

   k_1 &= f(\phi_n, y_n), \\
   k_2 &= f\left(\phi_n + \frac{h}{2}, y_n + \frac{h}{2}k_1\right), \\
   k_3 &= f\left(\phi_n + \frac{h}{2}, y_n + \frac{h}{2}k_2\right), \\
   k_4 &= f(\phi_n + h, y_n + h k_3), \\
   y_{n+1} &= y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4).

The actual step size is

.. math::

   h = \frac{2\pi}{n_{\mathrm{fp}}\,n_{\mathrm{step,per}}},

so ``nstep_per`` is the direct resolution knob per field period.

The integration horizon is controlled by:

- ``nstep_min``: minimum number of field periods before convergence is allowed
- ``nstep_max``: hard cap on the number of field periods
- ``acc_req``: tolerance used in rational-surface logic and stopping decisions

Rational surfaces and workload control
--------------------------------------

Near rational surfaces, the solver may need to traverse many field periods and
many starting angles. The characteristic scaling is

.. math::

   n_{\mathrm{fp,rat}}
   \sim
   \left\lceil\frac{1}{\mathrm{acc\_req}\,|\iota|}\right\rceil.

This is the source of the expensive low-``|iota|`` regime. NEO_JAX therefore
adds a preflight estimate before entering the costly correction:

- ``max_rational_field_periods`` limits the estimated field-period count
- ``rational_surface_policy="error"`` fails fast with a detailed diagnostic
- ``rational_surface_policy="approximate"`` keeps the base integration result
  and skips the expensive correction when the estimate is too large
- ``max_rational_field_periods=0`` requests the full exact long run

The approximate mode is intentionally explicit in the diagnostics and progress
output so that downstream studies can decide whether the controlled fallback is
acceptable.

JAX implementation strategy
---------------------------

The JAX backend leverages three main ideas:

- **precompute once per surface**:
  Fourier reconstruction and spline coefficients are built before the dominant
  integration loop
- **device-resident stepping**:
  the scan backend keeps RK4 staging and trapped-particle bookkeeping inside a
  compiled JAX loop
- **batched surface execution**:
  ``jax_surface_scan=True`` allows a compiled surface batch when the workflow is
  compatible with it

The JAX backend is most effective when:

- surface counts and grid sizes are reused across calls
- diagnostic file output is disabled
- the same compiled kernel is reused inside outer loops

Radial coordinates
------------------

NEO_JAX reports multiple radial coordinates derived from the Boozer flux grid:

- ``s``: normalized toroidal flux, defined on the Boozer surfaces as
  ``s = (j - 1.5) / (ns_b - 1)`` for surface index ``j`` (NEO convention).
- ``sqrt_s``: square root of ``s``, often used as a proxy for minor radius.
- ``r_eff``: effective radius computed by integrating the NEO quantity
  ``dr/dψ`` across the flux grid:

.. math::

   r_\mathrm{eff}(s_k) = \sum_{i=1}^{k} \left( \frac{dr}{d\psi} \right)_i \Delta s_i

This mirrors the reference NEO accumulation in the Fortran code.

Scaling and reported output
---------------------------

The internal integrals are converted to the reported output through the chosen
reference scaling:

- ``ref_swi = 1``:
  use the inner-grid reference field
- ``ref_swi = 2``:
  use the flux-surface maximum field

This affects the final reported ``b_ref`` and the scaling of
:math:`\epsilon_{\mathrm{eff}}^{3/2}`.
