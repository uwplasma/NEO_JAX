Overview
========

NEO_JAX is a JAX port of the STELLOPT NEO code, whose primary purpose is to
compute the effective helical ripple :math:`\epsilon_{\mathrm{eff}}^{3/2}` for
stellarator neoclassical transport in the :math:`1/\nu` regime. :cite:`nemov1999,stelopt-neo-docs`

The original NEO workflow uses Boozer-coordinate Fourier data produced by a
Boozer transform of VMEC equilibria and is documented in STELLOPT's NEO manual
and tutorial. :cite:`stelopt-neo-docs,stelopt-neo-tutorial`

NEO_JAX extends this workflow with an end-to-end differentiable pipeline that
can consume JAX-native VMEC and Boozer transforms, avoiding intermediate files
while remaining compatible with the legacy ``boozmn`` input format. :cite:`vmec-jax,booz-xform-jax`

Goals for this port include:

- Match ``xneo`` output to numerical tolerance on reference cases.
- Preserve the NEO algorithmic structure to support validation and debugging.
- Provide a JAX-friendly backend for just-in-time compilation and autodiff.
- Offer a single Python driver that composes VMEC, Boozer transform, and NEO.
