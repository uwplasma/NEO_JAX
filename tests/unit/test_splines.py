import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax.splines import eva2d, poi2d, spfper, spl2d, splper, splreg
from neo_jax.surface import build_splines


def test_splreg_reproduces_linear():
    x = jnp.linspace(0.0, 1.0, 6)
    y = 2.5 * x + 1.0
    h = float(x[1] - x[0])
    bi, ci, di = splreg(y, h)
    # A linear function should have zero curvature.
    assert np.allclose(np.asarray(ci[:-1]), 0.0, atol=1e-10)
    assert np.allclose(np.asarray(di[:-1]), 0.0, atol=1e-10)


@pytest.mark.parametrize("n", [3, 4, 9])
def test_splreg_is_the_natural_cubic_spline_and_its_jacobian(n):
    """splreg.f90 solves the natural end conditions S''(x_0) = S''(x_n) = 0 (de Boor 2001,
    ch. IV): pieces b, c, d and the Jacobian with respect to the data equal SciPy's."""
    from scipy.interpolate import CubicSpline

    x = np.linspace(0.2, 1.7, n)
    y = np.sin(3.0 * x) + x**3
    h = float(x[1] - x[0])

    def pieces(data):
        return jnp.stack(splreg(data, h))

    reference = CubicSpline(x, np.column_stack([y, np.eye(n)]), bc_type="natural").c
    expected = np.zeros((3, n, n + 1))
    expected[:, :-1] = reference[2::-1]
    np.testing.assert_allclose(pieces(jnp.asarray(y)), expected[..., 0], rtol=1e-12, atol=1e-12)
    jacobian = jax.jacfwd(pieces)(jnp.asarray(y))
    np.testing.assert_allclose(jacobian, expected[..., 1:], rtol=1e-12, atol=1e-12)


def test_splper_periodic_boundary():
    x = jnp.linspace(0.0, 2.0 * np.pi, 9)
    y = jnp.sin(x)
    h = float(x[1] - x[0])
    bi, ci, di = splper(y, h)
    # Periodic boundary conditions enforce coefficient wrap.
    assert np.isclose(np.asarray(bi[-1]), np.asarray(bi[0]))
    assert np.isclose(np.asarray(ci[-1]), np.asarray(ci[0]))
    assert np.isclose(np.asarray(di[-1]), np.asarray(di[0]))


def test_spl2d_and_eva2d_matches_grid():
    # Build a simple separable function on a grid.
    nx, ny = 8, 7
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    f = np.sin(2.0 * np.pi * xx) + np.cos(2.0 * np.pi * yy)

    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])

    spl = spl2d(jnp.asarray(f), hx, hy, mx=1, my=1)
    # Evaluate at grid points using poi2d + eva2d.
    for i in range(nx):
        for j in range(ny):
            ix, iy, dx, dy, ierr = poi2d(hx, hy, 1, 1, x[0], x[-1], y[0], y[-1], x[i], y[j])
            assert ierr == 0
            val = eva2d(spl, ix, iy, dx, dy)
            assert np.isclose(np.asarray(val), f[i, j], atol=1e-7)


@pytest.mark.parametrize("n", [2, 3, 4, 9, 33])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_periodic_factorization(n, dtype):
    a, b, c = map(np.asarray, spfper(n+1, dtype))
    factor = np.diag(a[:n])
    factor[np.arange(1, n-1), np.arange(n-2)] = b[:n-2]
    factor[-1, :-1] = c[:n-1]
    expected = 4*np.eye(n)
    expected[np.arange(n-1), np.arange(1, n)] = 1
    expected += np.triu(expected, 1).T
    expected[0, -1] = expected[-1, 0] = 2 if n == 2 else 1
    np.testing.assert_allclose(factor@factor.T, expected, atol=1e-6 if dtype == jnp.float32 else 1e-14)


@pytest.mark.parametrize("periodic,current", [(0, False), (1, True)])
def test_batched_splines_and_derivatives(periodic, current):
    names = ["b", "sqrg11", "kg", "pard"] + (["bqtphi"] if current else [])
    keys = ["b_spl", "g_spl", "k_spl", "p_spl"] + (["q_spl"] if current else [])
    fields = {name: jnp.sin(jnp.arange(63).reshape(9, 7)*(.01+i*.02)) for i, name in enumerate(names)}
    solve = jax.jit(lambda f: build_splines(f, .1, .2, periodic, periodic, current))
    actual, tangent = jax.jvp(solve, (fields,), (fields,))
    for name, key in zip(names, keys):
        expected = spl2d(fields[name], .1, .2, periodic, periodic)
        np.testing.assert_allclose(actual[key], expected, atol=1e-12)
        np.testing.assert_allclose(tangent[key], actual[key], atol=1e-12)


def _scipy_tensor_spline(x, y, f, px, py, bc):
    """Tensor-product cubic spline of SciPy (fit along x, then along y at each point) and
    its partials (i, j), i + j <= 2: the same interpolant as spl2d, by an independent code."""
    from scipy.interpolate import CubicSpline

    values = {}
    for i in range(3):
        along_x = CubicSpline(x, f, bc_type=bc)(px, i)  # (points, ny)
        spline_y = CubicSpline(y, along_x.T, bc_type=bc)
        for j in range(3 - i):
            values[i, j] = np.diagonal(spline_y(py, j), axis1=0, axis2=1)
    return values


@pytest.mark.parametrize("periodic", [1, 0])
@pytest.mark.parametrize("n", [17, 65])
def test_surface_splines_and_partials_match_scipy(periodic, n):
    """neo_eval and neo_bderiv on the splines of build_splines are the tensor-product cubic
    spline (periodic or natural on both axes) and its partials up to second order.

    The bound for a partial of order k is 100 eps h_min^-k times the size of the field:
    rounding of the knot data enters the interval polynomial divided by h^k. The two spline
    codes measured against each other differ by at most 1.5e-13 of that size for k = 2 on
    the 65^2 grid (bound 2e-10), and by the same at fewer points on the 17^2 grid."""
    from neo_jax.geometry import neo_bderiv, neo_eval
    from neo_jax.grids import prepare_grids

    nfp = 3
    grid = prepare_grids(n, n, nfp)
    grid = {**grid, "mt": periodic, "mp": periodic}
    x, y = np.asarray(grid["theta_arr"]), np.asarray(grid["phi_arr"])
    tt, pp = np.meshgrid(x, y, indexing="ij")
    fields = {}
    for k, name in enumerate(["b", "sqrg11", "kg", "pard"]):
        v = np.exp(0.3*np.sin((k+1)*tt - nfp*pp) + 0.2*np.cos(tt + 2*nfp*pp))
        if periodic:  # the seam rows equal to the last digit, as SciPy requires
            v[-1], v[:, -1] = v[0], v[:, 0]
        fields[name] = v + (0 if periodic else 0.1*k*tt)
    splines = build_splines({k: jnp.asarray(v) for k, v in fields.items()},
                            grid["theta_int"], grid["phi_int"], periodic, periodic)
    rng = np.random.default_rng(1)
    px, py = rng.uniform(0, x[-1], 200), rng.uniform(0, y[-1], 200)
    order = [splines[k] for k in ("b_spl", "g_spl", "k_spl", "p_spl")]
    evaluated = jax.vmap(lambda a, b: neo_eval(a, b, *order, None, grid)[:4])(px, py)
    partials = jax.vmap(lambda a, b: neo_bderiv(a, b, splines["b_spl"], grid))(px, py)
    bc = "periodic" if periodic else "natural"
    expected = {name: _scipy_tensor_spline(x, y, v, px, py, bc) for name, v in fields.items()}
    h = min(float(grid["theta_int"]), float(grid["phi_int"]))
    eps = np.finfo(np.float64).eps

    def check(actual, reference, k):
        scale = np.max(np.abs(expected["b"][0, 0]))
        np.testing.assert_allclose(actual, reference, rtol=0, atol=100*eps*h**-k*scale)

    for value, name in zip(evaluated, ["b", "sqrg11", "kg", "pard"]):
        check(value, expected[name][0, 0], 0)
    b = expected["b"]
    for value, key in zip(partials, [(1, 0), (0, 1), (2, 0), (1, 1), (1, 1), (0, 2)]):
        check(value, b[key], sum(key))


# Values of the unchanged spline code on ORBITS_FAST surfaces 64 and 96, 17 x 17 grid,
# FlintParams as below (eps_eff, b_ref) and the tangents with respect to scaling rmnc
# and bmnc by 1 + 0.01 t. Regression pins: the bounds are those of the existing check
# that two schedules of one kernel agree (rtol 1e-10 values, 1e-8 tangents); the
# splines themselves change only at rounding level (see the test above).
ORBITS_FAST_EPS_EFF = [5.0285664207883786e-06, 3.6841078062355706e-06]
ORBITS_FAST_B_REF = [5.3446474741365595, 5.431710238516683]
ORBITS_FAST_TANGENTS = [
    [-5.039611906679214e-08, -3.695846464428758e-08, 0.0, 0.0],
    [-1.0057132841576846e-07, -7.3682156124710707e-08, 5.3446474741365574e-02, 5.4317102385166835e-02],
]


def test_effective_ripple_and_its_derivatives_are_pinned():
    from dataclasses import replace
    from pathlib import Path

    from neo_jax.api import load_boozmn
    from neo_jax.driver import _solve_surfaces
    from neo_jax.integrate import FlintParams

    root = Path(__file__).resolve().parents[1] / "fixtures" / "orbits"
    data = load_boozmn(root / "boozmn_ORBITS_FAST.nc", surfaces=[64, 96])
    params = FlintParams(npart=8, multra=1, nstep_per=8, nstep_min=20, nstep_max=40,
                         acc_req=0.02, no_bins=8, calc_nstep_max=0)
    kernel = jax.jit(_solve_surfaces.__wrapped__, static_argnums=(2, 3, 4, 5, 6))
    indices = jnp.arange(2, dtype=jnp.int32)

    def objective(t):
        booz = replace(data, rmnc=data.rmnc*(1 + 0.01*t[0]), bmnc=data.bmnc*(1 + 0.01*t[1]))
        values = kernel(booz, indices, params, 17, 17, 2, None)
        return jnp.concatenate([values[0], values[-2]])

    value = objective(jnp.zeros(2))
    np.testing.assert_allclose(value, ORBITS_FAST_EPS_EFF + ORBITS_FAST_B_REF, rtol=1e-10)
    for k, expected in enumerate(ORBITS_FAST_TANGENTS):
        direction = jnp.zeros(2).at[k].set(1.0)
        _, tangent = jax.jvp(objective, (jnp.zeros(2),), (direction,))
        np.testing.assert_allclose(tangent, expected, rtol=1e-8, atol=1e-20)
