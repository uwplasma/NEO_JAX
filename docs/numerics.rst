Numerics
========

Grid and Fourier reconstruction
-------------------------------

NEO evaluates geometric quantities on a :math:`(\theta, \phi)` grid with
``theta_n`` and ``phi_n`` points. The toroidal angle spans one field period,
:math:`0 \le \phi \le 2\pi / n_{\mathrm{fp}}`. Fourier sums of the Boozer
coefficients are used to reconstruct :math:`R(\theta,\phi)`,
:math:`Z(\theta,\phi)`, the Boozer angle shift, and :math:`B(\theta,\phi)`.
Derived quantities (metric elements, :math:`|\nabla\psi|`, geodesic curvature,
and parallel derivative of :math:`B`) follow the original NEO formulas.

Spline interpolation
--------------------

The reconstructed fields are interpolated on the grid using periodic cubic
splines. The spline arrays mirror NEO's ``splper`` and ``spl2d`` routines and
are evaluated by ``neo_eval`` for fast ODE integration. :cite:`stelopt-neo-docs`

Field-line integration
----------------------

NEO advances field lines with a fixed-step RK4 integrator, taking ``nstep_per``
steps per field period. Accuracy and runtime are controlled by ``nstep_min``,
``nstep_max``, and the required accuracy ``acc_req``. :cite:`stelopt-neo-docs`

Near rational surfaces, NEO increases the number of field periods and averages
over multiple starting poloidal angles, using ``no_bins`` to ensure coverage of
poloidal bins. This logic is reproduced in the Python backend and will be added
to the JAX scan backend.

Diagnostics and output
----------------------

Optional diagnostics can be written when ``write_output_files``,
``write_integrate``, or ``write_diagnostic`` are enabled in the control file.
These files are useful for validating intermediate quantities against the
Fortran reference outputs. :cite:`stelopt-neo-docs,stelopt-neo-tutorial`
