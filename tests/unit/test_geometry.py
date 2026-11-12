import numpy as np
import pytest
import jax.numpy as jnp

from neo_jax.geometry import neo_zeros2d
from neo_jax.splines import spl2d


def test_neo_zeros2d_finds_quadratic_minimum():
    # Build a smooth quadratic function on a grid.
    nx, ny = 21, 23
    x = np.linspace(-1.0, 1.0, nx)
    y = np.linspace(-1.0, 1.0, ny)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    f = (xx - 0.3) ** 2 + (yy + 0.2) ** 2

    hx = float(x[1] - x[0])
    hy = float(y[1] - y[0])

    b_spl = spl2d(jnp.asarray(f), hx, hy, mx=0, my=0)

    grid = {
        "theta_int": hx,
        "phi_int": hy,
        "mt": 0,
        "mp": 0,
        "theta_start": float(x[0]),
        "theta_end": float(x[-1]),
        "phi_start": float(y[0]),
        "phi_end": float(y[-1]),
    }

    theta0 = jnp.array(0.0)
    phi0 = jnp.array(0.0)
    theta_f, phi_f, it, err = neo_zeros2d(theta0, phi0, 1e-10, 50, b_spl, grid)

    assert int(err) == 0
    assert np.isclose(np.asarray(theta_f), 0.3, atol=5e-3)
    assert np.isclose(np.asarray(phi_f), -0.2, atol=5e-3)


@pytest.mark.parametrize("amplitude", [0.0, 0.2])
def test_neo_zeros2d_phi_invariant(amplitude):
    angle = np.linspace(0, 2 * np.pi, 33)
    field = np.broadcast_to((2 + amplitude * np.cos(angle))[:, None], (33, 33))
    step = angle[1] - angle[0]
    spline = spl2d(jnp.asarray(field), step, step, mx=1, my=1)
    grid = dict(theta_int=step, phi_int=step, mt=1, mp=1,
                theta_start=0.0, theta_end=2*np.pi, phi_start=0.0, phi_end=2*np.pi)
    theta, phi, _, error = neo_zeros2d(jnp.array(2.8), jnp.array(0.7), 1e-10, 50,
                                     spline, grid, phi_invariant=True)
    assert int(error) == 0
    assert float(phi) == 0.7
    np.testing.assert_allclose(theta, np.pi if amplitude else 2.8, atol=1e-8)


def test_reverse_extremum_moves_with_field():
    import jax

    x, y = jnp.linspace(-1., 1., 11), jnp.linspace(-1., 1., 13)
    xx, yy = jnp.meshgrid(x, y, indexing='ij')
    hx, hy = float(x[1] - x[0]), float(y[1] - y[0])
    grid = {"theta_int": hx, "phi_int": hy, "mt": 0, "mp": 0,
            "theta_start": -1., "theta_end": 1., "phi_start": -1., "phi_end": 1.}

    def root(center, reverse):
        spline = spl2d((xx-center)**2 + (yy+.2)**2, hx, hy, mx=0, my=0)
        return neo_zeros2d(jnp.array(0.), jnp.array(0.), 1e-10, 50,
                           spline, grid, reverse_mode=reverse)[0]

    center = jnp.array(.3)
    value, derivative = jax.jit(jax.value_and_grad(lambda c: root(c, True)))(center)
    expected, tangent = jax.jvp(lambda c: root(c, False), (center,), (jnp.array(1.),))
    np.testing.assert_allclose(value, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(derivative, tangent, rtol=1e-9, atol=1e-10)
    step = 1e-5
    finite_difference = (root(center + step, False) - root(center - step, False)) / (2*step)
    assert float(derivative) > .9
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-6, atol=1e-8)
