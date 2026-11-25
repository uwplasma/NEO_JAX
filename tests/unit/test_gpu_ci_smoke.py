"""Small default-backend checks. Marked gpu so the GPU CI lane can select them;
they also pass on CPU."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax.integrate import flint_bo_jax
from neo_jax.geometry import neo_bderiv
from neo_jax.surface import build_splines
from test_flint_bo_jax_smoke import _constant_case

pytestmark = pytest.mark.gpu


def test_default_backend_array_is_float64():
    x = jnp.linspace(0.0, 1.0, 8)
    assert x.dtype == jnp.float64
    assert float(jnp.sum(x)) == pytest.approx(4.0)
    assert x.devices() == {jax.devices()[0]}


def test_surface_spline_on_default_backend():
    step = 2*np.pi/8
    angle = step*jnp.arange(9)
    b = jnp.broadcast_to(jnp.sin(angle)[:, None], (9, 9))
    spline = build_splines({k: b for k in ("b", "sqrg11", "kg", "pard")}, step, step, 1, 1)
    grid = dict(theta_int=step, phi_int=step, mt=1, mp=1, theta_start=0.0, phi_start=0.0)
    f, g, *_ = neo_bderiv(jnp.array(0.0), jnp.array(0.3), spline["b_spl"], grid)
    assert spline["b_spl"].devices() == {jax.devices()[0]}
    assert abs(float(f) - 1.0) < 1e-2 and abs(float(g)) < 1e-12


def test_flint_bo_jax_constant_field_on_default_backend():
    surface, params, env = _constant_case()
    out = flint_bo_jax(surface, params, env, nfp=1, Rmajor=1.0)
    assert np.isfinite(np.asarray(out["epstot"]))
    assert int(out["hit_rat"]) in (0, 1)
