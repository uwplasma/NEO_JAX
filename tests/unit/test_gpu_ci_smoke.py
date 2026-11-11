"""Small default-backend checks. Marked gpu so the GPU CI lane can select them;
they also pass on CPU."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax.integrate import flint_bo_jax
from neo_jax.splines import splper, splreg
from test_flint_bo_jax_smoke import _constant_case

pytestmark = pytest.mark.gpu


def test_default_backend_array_is_float64():
    x = jnp.linspace(0.0, 1.0, 8)
    assert x.dtype == jnp.float64
    assert float(jnp.sum(x)) == pytest.approx(4.0)
    assert x.devices() == {jax.devices()[0]}


def test_splreg_linear_on_default_backend():
    x = jnp.linspace(0.0, 1.0, 6)
    _, ci, di = splreg(2.5 * x + 1.0, float(x[1] - x[0]))
    assert np.allclose(np.asarray(ci[:-1]), 0.0, atol=1e-10)
    assert np.allclose(np.asarray(di[:-1]), 0.0, atol=1e-10)


def test_splper_periodic_on_default_backend():
    x = jnp.linspace(0.0, 2.0 * np.pi, 9)
    bi, ci, _ = splper(jnp.sin(x), float(x[1] - x[0]))
    assert np.isclose(np.asarray(bi[-1]), np.asarray(bi[0]))
    assert np.isclose(np.asarray(ci[-1]), np.asarray(ci[0]))


def test_flint_bo_jax_constant_field_on_default_backend():
    surface, params, env = _constant_case()
    out = flint_bo_jax(surface, params, env, nfp=1, Rmajor=1.0)
    assert np.isfinite(np.asarray(out["epstot"]))
    assert int(out["hit_rat"]) in (0, 1)
