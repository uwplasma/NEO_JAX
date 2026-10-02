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
