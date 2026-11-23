"""Bounce-integral effective ripple: analytic anchors, cross-code references, gradients, convergence."""

from pathlib import Path

import jax
import numpy as np

from neo_jax import BoozerData, RippleBudget, booz_xform_to_boozerdata_jax, effective_ripple_3_2

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "ripple"
NFP = 2


def _torus(modes, iota=0.41, current=0.0, r=0.2, R0=5.0, shift=None):
    """Large-aspect-ratio Boozer surface, |B| = 1 + sum a cos(m theta - n phi), circular section.

    ``shift = (c, d)`` rotates the Boozer angles, which produces the sine partners of every table.
    """
    xm = np.array([0, 1] + [m for m, _, _ in modes], float)
    xn = np.array([0, 0] + [n for _, n, _ in modes], float)
    b, rm, zs = np.zeros(xm.size), np.zeros(xm.size), np.zeros(xm.size)
    b[0], rm[0], rm[1], zs[1] = 1.0, R0, r, r
    b[2:] = [a for _, _, a in modes]
    zero = np.zeros((1, xm.size))
    kwargs = {}
    if shift is not None:
        delta = xm * shift[0] - xn * shift[1]
        c, s = np.cos(delta), np.sin(delta)
        kwargs = dict(bmns=(-b * s)[None], rmns=(-rm * s)[None], zmnc=(zs * s)[None])
        b, rm, zs = b * c, rm * c, zs * c
    return BoozerData(rmnc=rm[None], zmns=zs[None], lmns=zero, bmnc=b[None], ixm=xm, ixn=xn, es=np.array([0.5]),
                      iota=np.array([iota]), curr_pol=np.array([R0]), curr_tor=np.array([current]), nfp=NFP,
                      **({"lmnc": zero, **kwargs} if kwargs else {}))


def _eps(booz, budget=None):
    return float(effective_ripple_3_2(booz, budget)[0])


def _fixture(name):
    with np.load(FIXTURES / name) as data:
        return booz_xform_to_boozerdata_jax({**{k: data[k] for k in data.files}, "mode_first": True}, mode_first=True)


def test_quasi_symmetric_fields_have_no_ripple_and_a_perturbation_scales_quadratically():
    """A single helicity (QA or QH, with a net current) gives zero: I_1 is the integral of an exact derivative
    between bounce points (Nemov et al. 1999). A perturbation of relative size lambda gives lambda^2."""
    base = _eps(_torus([(1, 0, -0.1), (1, NFP, -0.02)]))
    for modes, current in (([(1, 0, -0.1)], 0.0), ([(1, NFP, -0.1)], 0.3)):
        assert _eps(_torus(modes, current=current)) < 1e-8 * base
    values = [_eps(_torus([(1, 0, -0.1), (1, NFP, -0.005 * lam)])) / lam**2 for lam in (1, 2)]
    np.testing.assert_allclose(values[1], values[0], rtol=0.05)


def test_value_is_invariant_to_angle_shifts_and_field_scale():
    modes = [(1, 0, -0.1), (1, NFP, -0.03), (2, NFP, 0.01)]
    reference = _eps(_torus(modes, current=0.2))
    np.testing.assert_allclose(_eps(_torus(modes, current=0.2, shift=(0.7, 0.4))), reference, rtol=2e-2)
    scaled = _torus(modes, current=0.2)
    scaled = BoozerData(**{**scaled.__dict__, "bmnc": 2.5 * scaled.bmnc, "curr_pol": 2.5 * scaled.curr_pol,
                           "curr_tor": 2.5 * scaled.curr_tor})
    np.testing.assert_allclose(_eps(scaled), reference, rtol=1e-6)


def test_matches_fortran_xneo_neo_jax_and_desc_and_converges_in_the_budget():
    """QI spectrum (nfp 4, iota ~ -1.3, finite beta): xneo (128 x 128, 96 particles, 80 steps per period,
    acc_req 1e-3; NaN at the third surface, where the 64 x 64 run gives 4.5405e-4) and NEO_JAX at that
    resolution.  Vacuum nfp-2 ellipse: xneo (NaN at s = 0.25, so its 64 x 64 run's 1.1828e-2) and DESC
    (L=16, M=12, Y_B=192, 100 transits).  DESC differs by 30% on the QI spectrum for a reason not found
    (``docs/validation.rst``), so it is compared on the ellipse only."""
    qi = _fixture("boozer_nfp4_QI_finite_beta.npz")
    values = np.asarray(effective_ripple_3_2(qi))
    np.testing.assert_allclose(values, [5.1933e-4, 4.7918e-4, 4.5405e-4], rtol=0.03)
    np.testing.assert_allclose(values, [5.2224e-4, 4.7922e-4, 4.5097e-4], rtol=0.03)
    ref = np.asarray(effective_ripple_3_2(qi, RippleBudget(48, 6, 6, 48, 8, 64)))
    np.testing.assert_allclose(ref[1], 4.7918e-4, rtol=3e-3)
    np.testing.assert_allclose(values, ref, rtol=0.01)
    ellipse = np.asarray(effective_ripple_3_2(_fixture("boozer_ellipse_nfp2.npz")))
    np.testing.assert_allclose(ellipse, [1.1828e-2, 1.6702e-2, 2.1461e-2], rtol=0.02)
    np.testing.assert_allclose(ellipse, [1.2128e-2, 1.6863e-2, 2.1762e-2], rtol=0.03)


def test_gradient_forward_reverse_and_finite_difference_agree():
    modes = [(1, 0, -0.1), (1, NFP, -0.03), (2, NFP, 0.01)]
    booz = _torus(modes, current=0.2)
    budget = RippleBudget(n_lines=16, n_well=32)

    def f(a, iota, current):
        return effective_ripple_3_2(BoozerData(**{**booz.__dict__, "bmnc": a[None], "iota": iota[None], "curr_tor": current[None]}), budget)[0]

    x0 = (jax.numpy.asarray(booz.bmnc[0]), jax.numpy.asarray(0.41), jax.numpy.asarray(0.2))
    grads = jax.jit(jax.grad(f, argnums=(0, 1, 2)))(*x0)
    direction = (jax.numpy.asarray(np.random.default_rng(1).normal(size=x0[0].size)) * 1e-2, 1.0, 1.0)
    forward = jax.jvp(f, x0, direction)[1]
    reverse = sum(float(np.vdot(g, t)) for g, t in zip(grads, direction))
    np.testing.assert_allclose(forward, reverse, rtol=1e-9)
    h = 1e-5
    central = (f(*[x + h * t for x, t in zip(x0, direction)]) - f(*[x - h * t for x, t in zip(x0, direction)])) / (2 * h)
    np.testing.assert_allclose(central, reverse, rtol=1e-4)
