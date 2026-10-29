"""A real ripple derivative, including geometry, field, iota and rational replay."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax import NeoConfig
from neo_jax.data_models import BoozerData
from neo_jax.driver import run_neo_from_boozer_jax


def _toy_boozer():
    return BoozerData(
        ixm=jnp.array([0, 0, 1, 1, 1, 2]),
        ixn=jnp.array([0, 2, 0, 2, -2, 0]),
        nfp=2,
        es=jnp.array([0.5]),
        iota=jnp.array([0.43]),
        curr_pol=jnp.array([10.0]),
        curr_tor=jnp.array([0.1]),
        **{
            k: jnp.array(v)[None]
            for k, v in {
                "rmnc": [10.0, 0.2, 1.0, 0.1, 0.05, 0.02],
                "zmns": [0.0, 0.1, 1.0, 0.05, -0.03, 0.01],
                "lmns": [0.0, 0.02, 0.03, 0.01, 0.0, 0.0],
                "bmnc": [1.0, 0.03, -0.1, 0.04, 0.02, 0.01],
            }.items()
        },
    )


@pytest.mark.parametrize("rational", [False, True])
def test_ripple_reverse_matches_forward_and_fd(rational):
    data = _toy_boozer()
    if rational:
        data = replace(data, iota=jnp.array([0.501]))
    cfg = NeoConfig(
        theta_n=8,
        phi_n=8,
        npart=6,
        multra=1,
        nstep_per=4,
        nstep_min=4,
        nstep_max=8,
        acc_req=0.2,
        no_bins=2,
        calc_nstep_max=0 if rational else 1,
        max_rational_field_periods=0,
    )

    def objective(x, reverse):
        booz = replace(
            data,
            bmnc=data.bmnc.at[0, 3].add(0.01 * x[0]),
            rmnc=data.rmnc.at[0, 2].add(0.01 * x[1]),
            iota=data.iota + 0.001 * x[2],
            curr_pol=data.curr_pol + 0.01 * x[3],
        )
        control = replace(cfg, reverse_mode=reverse).to_control()
        return run_neo_from_boozer_jax(booz, control, max_rational_field_periods=0).eps_eff.sum()

    if rational:
        # Force a real rational correction across the eight-step replay block.
        from neo_jax import build_surface_problem
        from neo_jax.integrate import flint_bo_jax

        problem = build_surface_problem(data, cfg, surface=1)
        result = jax.jit(
            lambda: flint_bo_jax(
                problem.surface, problem.params, problem.env, nfp=2, rt0=problem.Rmajor
            )
        )()
        assert int(result["hit_rat"]) == 1
        assert int(result["nfp_rat"]) > 8

    x, direction = jnp.zeros(4), jnp.array([0.4, -0.3, 0.2, 0.1])
    value, gradient = jax.jit(jax.value_and_grad(lambda x: objective(x, True)))(x)
    base, tangent = jax.jit(lambda x, d: jax.jvp(lambda x: objective(x, False), (x,), (d,)))(
        x, direction
    )
    step = 1e-4
    fd = (objective(x + step * direction, False) - objective(x - step * direction, False)) / (
        2 * step
    )
    np.testing.assert_allclose(value, base, rtol=1e-12, atol=1e-14)
    assert float(value) > 0 and np.all(np.isfinite(gradient))
    assert abs(float(tangent)) > 1e-10
    assert abs(float(gradient[0])) > 1e-10  # nonuniform magnetic spectrum
    assert abs(float(gradient[2])) > 1e-10  # physical iota sensitivity
    np.testing.assert_allclose(jnp.vdot(gradient, direction), tangent, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(jnp.vdot(gradient, direction), fd, rtol=1e-5, atol=1e-10)
