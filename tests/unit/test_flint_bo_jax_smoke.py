from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax.integrate import FlintParams, RhsEnv, flint_bo, flint_bo_jax
from neo_jax.surface import SurfaceData, build_splines


def _constant_case(iota=0.5):
    jax.config.update("jax_enable_x64", True)

    theta_n, phi_n = 6, 6
    theta_arr = jnp.linspace(0.0, 2.0 * np.pi, theta_n)
    phi_arr = jnp.linspace(0.0, 2.0 * np.pi, phi_n)

    b = jnp.ones((theta_n, phi_n)) * 2.0
    sqrg11 = jnp.ones((theta_n, phi_n)) * 3.0
    kg = jnp.ones((theta_n, phi_n)) * 4.0
    pard = jnp.ones((theta_n, phi_n)) * 0.1

    fields = {"b": b, "sqrg11": sqrg11, "kg": kg, "pard": pard}

    theta_int = float(theta_arr[1] - theta_arr[0])
    phi_int = float(phi_arr[1] - phi_arr[0])

    splines = build_splines(fields, theta_int, phi_int, mt=1, mp=1, calc_cur=False)

    surface = SurfaceData(
        b_min=jnp.array(2.0),
        b_max=jnp.array(2.0),
        theta_bmin=jnp.array(0.0),
        phi_bmin=jnp.array(0.0),
        theta_bmax=jnp.array(0.0),
        phi_bmax=jnp.array(0.0),
        bmref=jnp.array(2.0),
        fields=fields,
        splines=splines,
    )

    grid = {
        "theta_int": theta_int,
        "phi_int": phi_int,
        "theta_start": float(theta_arr[0]),
        "theta_end": float(theta_arr[-1]),
        "phi_start": float(phi_arr[0]),
        "phi_end": float(phi_arr[-1]),
        "mt": 1,
        "mp": 1,
    }

    params = FlintParams(
        npart=3,
        multra=1,
        nstep_per=1,
        nstep_min=0,
        nstep_max=2,
        acc_req=0.1,
        no_bins=4,
        calc_nstep_max=0,
    )

    env = RhsEnv(
        splines=surface.splines,
        grid=grid,
        eta=jnp.array([0.0]),
        bmod0=surface.bmref,
        iota=jnp.array(iota),
        curr_pol=jnp.array(1.0),
        curr_tor=jnp.array(0.0),
    )

    return surface, params, env


def test_flint_bo_jax_smoke_constant_field():
    surface, params, env = _constant_case()
    out = flint_bo_jax(surface, params, env, nfp=1, Rmajor=1.0)

    assert np.isfinite(np.asarray(out["epstot"]))
    assert int(out["hit_rat"]) in (0, 1)


@pytest.mark.parametrize("integrator", [flint_bo, flint_bo_jax])
@pytest.mark.parametrize("iota", [0.50001, -0.50001])
def test_rational_period_count_is_positive(integrator, iota):
    surface, params, env = _constant_case(iota)
    params = replace(params, nstep_min=2, nstep_max=5, acc_req=0.5, no_bins=2)
    out = integrator(surface, params, env, nfp=1, Rmajor=1.0)

    assert int(out["hit_rat"]) == 1
    assert int(out["nfp_rat"]) == 4
    assert int(out["final_n"]) == 8
    assert np.isfinite(np.asarray(out["epstot"]))


@pytest.mark.parametrize("integrator", [flint_bo, flint_bo_jax])
@pytest.mark.parametrize("iota", [0.5, -0.5])
def test_exact_rational_closure_applies_correction(integrator, iota):
    surface, params, env = _constant_case(iota)
    params = replace(params, nstep_min=2, nstep_max=5, acc_req=0.5, no_bins=2)
    out = integrator(surface, params, env, nfp=1, Rmajor=1.0)

    assert np.isinf(np.asarray(out["n_gap"]))
    assert int(out["hit_rat"]) == 1
    assert int(out["nfp_rat"]) == 4
    assert int(out["final_n"]) == 8
    assert bool(out["rational_correction_applied"])
    assert np.isfinite(np.asarray(out["epstot"]))


@pytest.mark.parametrize("integrator", [flint_bo, flint_bo_jax])
@pytest.mark.parametrize("iota", [0.50001, -0.50001])
def test_gap_estimation_mode_preserves_large_finite_diagnostic(integrator, iota):
    surface, params, env = _constant_case(iota)
    params = replace(params, nstep_min=2, nstep_max=5, acc_req=0.5, no_bins=2, calc_nstep_max=1)
    out = integrator(surface, params, env, nfp=1, Rmajor=1.0)

    assert np.isfinite(np.asarray(out["n_gap"]))
    assert float(out["n_gap"]) > 10_000
    assert int(out["hit_rat"]) == 0
    assert int(out["final_n"]) == 5
    assert not bool(out["rational_correction_applied"])


@pytest.mark.parametrize("integrator", [flint_bo, flint_bo_jax])
@pytest.mark.parametrize("iota", [0.50001, -0.50001, 0.0])
def test_skipped_correction_reports_only_executed_periods(integrator, iota):
    surface, params, env = _constant_case(iota)
    params = replace(params, nstep_min=2, nstep_max=5, acc_req=0.5, no_bins=2)
    out = integrator(surface, params, env, nfp=1, Rmajor=1.0, skip_rational_correction=True)

    assert int(out["hit_rat"]) == 1
    assert int(out["final_n"]) == int(out["nintfp"]) > 0
    assert not bool(out["rational_correction_applied"])
    assert np.isfinite(np.asarray(out["epstot"]))


def test_gap_decision_avoids_integer_overflow_without_truncating_estimate():
    from neo_jax.integrate import _rational_gap

    for dtype in (jnp.float32, jnp.float64):
        for residual, expected in [(0.0, np.inf), (1e-12, 2e12), (2**-10, 2048.0)]:
            gap, decision = _rational_gap(jnp.asarray(1.0, dtype), jnp.asarray(residual, dtype), 2, 5)
            np.testing.assert_allclose(np.asarray(gap), expected, rtol=1e-6)
            assert int(decision) == 6


@pytest.mark.parametrize("iota", [0.0, 1e-12, -1e-12, 2e-9])
def test_unrepresentable_full_correction_is_not_a_finite_result(iota):
    surface, params, env = _constant_case(iota)
    params = replace(params, nstep_min=2, nstep_max=5, acc_req=0.5, no_bins=2)
    with pytest.raises(ValueError, match="representable period counts"):
        flint_bo(surface, params, env, nfp=1, Rmajor=1.0)
    out = jax.jit(lambda e: flint_bo_jax(surface, params, e, nfp=1, Rmajor=1.0))(env)
    assert not bool(out["rational_correction_valid"])
    assert not bool(out["rational_correction_applied"])
    assert not np.isfinite(np.asarray(out["epstot"]))
    assert np.isnan(np.asarray(out["epspar"])).all()
    assert int(out["final_n"]) == int(out["nintfp"]) > 0
    skipped = flint_bo_jax(surface, params, env, nfp=1, Rmajor=1.0, skip_rational_correction=True)
    assert bool(skipped["rational_correction_valid"])
    assert not bool(skipped["rational_correction_applied"])
    assert np.isfinite(np.asarray(skipped["epstot"]))


def test_rounded_and_total_correction_counts_checked_before_int32_conversion():
    from neo_jax.integrate import _rational_periods

    # Rounding to a rational denominator and multiplying by the number of
    # starting lines can overflow even when the initial period count fits.
    for dtype in (jnp.float32, jnp.float64):
        for iota, denominator, bins in [(2.0/(2**31-1), 2, 0), (2e-9, 1, 2)]:
            periods, lines = _rational_periods(jnp.asarray(iota, dtype), 0.5, denominator, bins)
            assert int(periods) == int(lines) == 0
        periods, lines = _rational_periods(jnp.asarray(2**-29, dtype), 0.5, 1, 0)
        assert int(periods) == 2**30
        assert int(lines) == 0


def test_public_driver_preserves_invalid_rational_result():
    from pathlib import Path
    from neo_jax import NeoConfig, run_neo
    from neo_jax.api import load_boozmn

    path = Path(__file__).resolve().parents[1]/"fixtures/orbits/boozmn_ORBITS_FAST.nc"
    booz = load_boozmn(path, surfaces=[64])
    booz = replace(booz, iota=np.zeros_like(booz.iota))
    config = NeoConfig(surfaces=[1], theta_n=8, phi_n=8, npart=3, multra=1,
                       nstep_per=1, nstep_min=2, nstep_max=5, acc_req=0.5,
                       no_bins=2, max_rational_field_periods=0)
    result = run_neo(booz, config=config, use_jax=True)[0]
    assert not bool(result.diagnostics["rational_correction_valid"])
    assert np.isnan(result.epsilon_effective)
