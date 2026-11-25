import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax.surface import build_splines


@pytest.mark.parametrize("periodic,current", [(0, False), (1, True)])
def test_batched_splines_and_their_tangents(periodic, current):
    """One batched fit equals the fit of each field alone, under jit, and is linear in the data."""
    from solvax import tensor_spline_coefficients

    names = ["b", "sqrg11", "kg", "pard"] + (["bqtphi"] if current else [])
    keys = ["b_spl", "g_spl", "k_spl", "p_spl"] + (["q_spl"] if current else [])
    fields = {name: jnp.sin(jnp.arange(63).reshape(9, 7)*(.01+i*.02)) for i, name in enumerate(names)}
    solve = jax.jit(lambda f: build_splines(f, .1, .2, periodic, periodic, current))
    actual, tangent = jax.jvp(solve, (fields,), (fields,))
    bc = ("periodic",)*2 if periodic else ("natural",)*2
    for name, key in zip(names, keys):
        expected = tensor_spline_coefficients(.1*jnp.arange(9), .2*jnp.arange(7), fields[name], bc=bc)
        scale = np.max(np.abs(expected), axis=(1, 3), keepdims=True)  # per power block
        np.testing.assert_allclose(actual[key]/scale, expected/scale, atol=1e-13)
        np.testing.assert_allclose(tangent[key]/scale, actual[key]/scale, atol=1e-13)


def _scipy_tensor_spline(x, y, f, px, py, bc):
    """Tensor-product cubic spline of SciPy (fit along x, then along y at each point) and
    its partials (i, j), i + j <= 2: the interpolant of build_splines, by an independent code."""
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
