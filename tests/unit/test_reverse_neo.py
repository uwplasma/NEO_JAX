"""A real ripple derivative, including geometry, field, iota and rational replay."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax import NeoConfig, run_neo
from neo_jax.data_models import BoozerData


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
    )

    def objective(x, reverse):
        booz = replace(
            data,
            bmnc=data.bmnc.at[0, 3].add(0.01 * x[0]),
            rmnc=data.rmnc.at[0, 2].add(0.01 * x[1]),
            iota=data.iota + 0.001 * x[2],
            curr_pol=data.curr_pol + 0.01 * x[3],
        )
        mapping = {name + "_b": getattr(booz, name) for name in ("rmnc", "zmns", "bmnc")}
        mapping.update(
            nfp_b=booz.nfp,
            ixm_b=booz.ixm,
            ixn_b=booz.ixn,
            pmns_b=-booz.lmns * (2 * jnp.pi / booz.nfp),
            s_b=booz.es,
            iota_b=booz.iota,
            buco_b=booz.curr_tor,
            bvco_b=booz.curr_pol,
        )
        return run_neo(
            mapping, config=replace(cfg, reverse_mode=reverse), use_jax=True, jax_surface_scan=True
        ).eps_eff.sum()

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


def test_traced_live_iota_guard_skips_integration(monkeypatch):
    from neo_jax import driver

    called = []

    def kernel(booz, indices, params, *args, **kwargs):
        jax.debug.callback(lambda: called.append(True), ordered=True)
        scalar = jnp.ones((len(indices),), dtype=booz.bmnc.dtype)
        return (scalar, scalar[:, None]) + (scalar,) * 11

    monkeypatch.setattr(driver, "_solve_surfaces", kernel)
    data = _toy_boozer()
    cfg = NeoConfig(max_rational_field_periods=200, multra=1)

    @jax.jit
    def solve(iota):
        return driver.run_neo_from_boozer_jax(
            replace(data, iota=iota[None]),
            cfg.to_control(),
            max_rational_field_periods=cfg.max_rational_field_periods,
        )

    rejected = solve(jnp.array(1e-8))
    rejected.eps_eff.block_until_ready()
    assert np.isnan(np.asarray(rejected.eps_eff)).all()
    assert bool(rejected.diagnostics["rational_work_limit_exceeded"][0])
    assert called == []
    accepted = solve(jnp.array(0.43))
    accepted.eps_eff.block_until_ready()
    assert not bool(accepted.diagnostics["rational_work_limit_exceeded"][0])
    assert called == [True]
    assert np.isfinite(np.asarray(accepted.eps_eff)).all()


def test_vmapped_live_guard_rejects_before_execution(monkeypatch):
    from neo_jax import driver

    data = _toy_boozer()
    cfg = NeoConfig(max_rational_field_periods=200, multra=1)

    def kernel(booz, indices, params, *args, **kwargs):
        scalar = jnp.ones((len(indices),), dtype=booz.bmnc.dtype)
        return (scalar, scalar[:, None]) + (scalar,) * 11

    monkeypatch.setattr(driver, "_solve_surfaces", kernel)

    @jax.jit
    def solve(iota):
        return driver.run_neo_from_boozer_jax(
            replace(data, iota=iota[None]),
            cfg.to_control(),
            max_rational_field_periods=cfg.max_rational_field_periods,
        ).eps_eff

    with pytest.raises(ValueError, match="jax.lax.map"):
        jax.vmap(solve)(jnp.array([0.43, 1e-8]))
