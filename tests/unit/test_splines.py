import numpy as np
import jax.numpy as jnp

from neo_jax.splines import eva2d, poi2d, spl2d, splper, splreg


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
