# Effective-ripple fixtures

Mode-first Boozer spectra (NEO_JAX mapping keys, `mode_first=True`) of two VMEX equilibria, used by
`tests/unit/test_ripple.py` with cross-code references:

- `boozer_nfp4_QI_finite_beta.npz`: nfp = 4 quasi-isodynamic, finite beta, iota ~ -1.3, s = 0.43, 0.49, 0.55
  (`mboz=16, nboz=12`).
- `boozer_ellipse_nfp2.npz`: nfp = 2 vacuum rotating ellipse, s = 0.25, 0.5167, 0.75 (`mboz=12, nboz=9`).

References: STELLOPT `xneo` (Fortran) with a 128 x 128 spline, 96 particles, 80 steps per period,
`acc_req = 1e-3`; NEO_JAX at the same controls; DESC (L=16, M=12, `Y_B=192`, 100 field-period transits,
64 quadrature nodes, 256 pitch nodes) on the vacuum ellipse only.
