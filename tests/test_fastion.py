"""Gamma_c / Gamma_alpha kernels on analytic Boozer fields: omnigenous limit, bounds, forward-mode vs finite differences."""

import jax
import jax.numpy as jnp
import numpy as np

from neo_jax.data_models import BoozerData
from neo_jax.fastion import FastIonBudget, fast_ion_proxies

jax.config.update("jax_enable_x64", True)

NFP = 2
ES = np.linspace(0.3, 0.7, 5)
BUDGET = FastIonBudget(n_lines=8, n_transits=2, nodes_per_period=16, n_panel=16, n_level=24, n_well=24, n_dense=48)


def _data(amps, xm, xn):
    b = jnp.stack([jnp.ones(5)] + [a * jnp.ones(5) for a in amps], 1)
    z = jnp.zeros_like(b)
    return BoozerData(rmnc=z, zmns=z, lmns=z, bmnc=b, ixm=np.array([0, *xm]), ixn=np.array([0, *xn]), es=jnp.asarray(ES),
                      iota=0.4 + 0.1 * jnp.asarray(ES), curr_pol=jnp.ones(5), curr_tor=jnp.zeros(5), nfp=NFP)


def test_quasi_helical_single_harmonic_is_omnigenous():
    """B = 1 + eps cos(theta - N phi) depends on alpha only through a phase: dJ/dalpha = 0, so both vanish up to the finite window and quadrature (measured 2.4e-4 and 1.4e-3 against 1e-2 and 0.2 for a rippled field)."""
    gc, ga, ft = fast_ion_proxies(_data([0.15], [1], [NFP]), BUDGET)
    assert float(gc[0]) < 5e-4 and float(ga[0]) < 3e-3 and 0.1 < float(ft[0]) < 0.6


def test_gamma_alpha_is_bounded_by_the_trapped_fraction():
    gc, ga, ft = fast_ion_proxies(_data([0.1, 0.03, 0.02], [1, 1, 2], [NFP, 0, NFP]), BUDGET)
    assert float(gc[0]) > 1e-3 and 0.0 <= float(ga[0]) <= float(ft[0]) + 1e-9


def test_forward_mode_matches_finite_differences():
    """``jvp`` against central differences: needs the roundoff-converged bounce roots of ``neo_jax.ripple`` (an unconverged root gave a derivative off by 100 %)."""
    def gamma_c(e):
        return fast_ion_proxies(_data([e, 0.03, 0.02], [1, 1, 2], [NFP, 0, NFP]), BUDGET)[0][0]

    e = 0.15
    jvp = float(jax.jvp(gamma_c, (e,), (1.0,))[1])
    fd = (float(gamma_c(e + 1e-3)) - float(gamma_c(e - 1e-3))) / 2e-3
    assert abs(jvp / fd - 1.0) < 0.03


def test_gamma_alpha_forward_mode_matches_finite_differences():
    """The ``jvp`` against a central difference at 3e-5 (agreement 2e-4).  At the 8-line test budget the value
    carries wiggles of 3e-5 (1e-4 relative), so steps of 1e-4 to 3e-4 give slopes 0.67 and 0.55 for 0.49."""
    def gamma_alpha(e):
        return fast_ion_proxies(_data([e, 0.03, 0.02], [1, 1, 2], [NFP, 0, NFP]), BUDGET)[1][0]

    e, h = 0.15, 3e-5
    jvp = float(jax.jvp(gamma_alpha, (e,), (1.0,))[1])
    fd = (float(gamma_alpha(e + h)) - float(gamma_alpha(e - h))) / (2 * h)
    assert abs(jvp / fd - 1.0) < 0.03


def test_gamma_alpha_is_smooth_in_the_field():
    """Nine amplitudes across 0.14-0.16: the second differences stay below 5 % of the change over the scan."""
    v = np.array([float(fast_ion_proxies(_data([e, 0.03, 0.02], [1, 1, 2], [NFP, 0, NFP]), BUDGET)[1][0])
                  for e in np.linspace(0.14, 0.16, 9)])
    assert np.abs(v[2:] - 2 * v[1:-1] + v[:-2]).max() < 0.05 * abs(v[-1] - v[0])


def test_drift_integrals_are_the_derivatives_of_the_action():
    """``N = dJ/dalpha / G^`` and ``D = dJ/ds / G^`` of one well against finite differences of
    ``J = G^ int sqrt(1 - B/b) dphi / B`` (scipy bounce points and quadrature, independent of the kernel).

    ``B = 1 + e1(s) cos(theta - 2 phi) + e2 cos theta``, ``iota = 0.4 + 0.1 s``, ``G^ = 1 + 0.3 s``; the
    derivative in ``s`` follows the field line through the fixed midpoint of the well.
    """
    from scipy.integrate import quad
    from scipy.optimize import brentq

    from neo_jax.fastion import _wells
    from neo_jax.ripple import _cubic, _peval, _series, _stationary

    nfp, nturn, text, nper, s0 = 2, 2, 1, 24, 0.5
    n, nstart = nfp * (nturn + text) * nper, nfp * nturn * nper
    L, h = 2 * np.pi * nturn, 2 * np.pi * (nturn + text) / n
    xm, xn = np.array([0.0, 1.0, 1.0]), np.array([0.0, 2.0, 0.0])
    e1 = lambda s: 0.10 + 0.2 * (s - 0.5)
    e2 = 0.04
    iota = lambda s: 0.4 + 0.1 * s
    Gh = lambda s: 1.0 + 0.3 * s
    B = lambda s, th, ph: 1 + e1(s) * np.cos(th - 2 * ph) + e2 * np.cos(th)
    amp = jnp.asarray([1.0, e1(s0), e2], dtype=complex)
    damp = jnp.asarray([0.0, 0.2, 0.0], dtype=complex)
    alpha, b = 0.7, 1.05
    phi = jnp.arange(n + 1) * h
    th = alpha + iota(s0) * phi
    Bn, Bt, Bp, Btt, Btp, _ = _series(amp, xm, xn, th, phi, order=2)
    Sn, St, Sp = _series(damp, xm, xn, th, phi)
    cB, cT, cS = (_cubic(Bn, Bp + iota(s0) * Bt, h), _cubic(Bt, Btp + iota(s0) * Btt, h),
                  _cubic(Sn, Sp + iota(s0) * St, h))
    ta, tb = _stationary(cB)
    brk = jnp.stack([jnp.zeros(n), ta, tb, jnp.ones(n)], -1)
    iv = jnp.repeat(jnp.arange(n), 3)
    t0, t1 = brk[:, :3].ravel(), brk[:, 1:].ravel()
    (win, vt, Nn, Dn, mid), _ = _wells((cB, cT, cS), (iv, t0, t1, _peval(cB[iv], t0)), h, b, iota(s0), 0.3 / Gh(s0),
                                       0.1, nstart, 0.25 * L, 8, 64, 1.0, 1.0)
    k = int(np.argmax(np.asarray(win) * (np.asarray(vt) > 0)))
    assert float(win[k]) > 0.9
    centre = float(mid[k])

    def action(a, s):
        f = lambda p: B(s, a + iota(s) * p, p) - b
        lo = centre
        while f(lo) < 0:
            lo -= 0.01
        hi = centre
        while f(hi) < 0:
            hi += 0.01
        p1, p2 = brentq(f, lo, lo + 0.01), brentq(f, hi - 0.01, hi)
        c, d = 0.5 * (p1 + p2), 0.5 * (p2 - p1)

        def g(u):
            p = c + d * np.sin(u)
            return np.sqrt(max(1 - B(s, a + iota(s) * p, p) / b, 0.0)) / B(s, a + iota(s) * p, p) * d * np.cos(u)

        return Gh(s) * quad(g, -np.pi / 2, np.pi / 2, epsabs=1e-13, epsrel=1e-13)[0]

    d = 1e-5
    dJa = (action(alpha + d, s0) - action(alpha - d, s0)) / (2 * d)
    theta_c = alpha + iota(s0) * centre
    along = lambda s: action(theta_c - iota(s) * centre, s)
    dJs = (along(s0 + d) - along(s0 - d)) / (2 * d)
    np.testing.assert_allclose(float(Nn[k]) * Gh(s0), dJa, rtol=1e-4, atol=1e-8)
    np.testing.assert_allclose(float(Dn[k]) * Gh(s0), dJs, rtol=1e-3, atol=1e-8)
