"""Regression coverage for low-iota constellaration cases that used to appear to hang."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neo_jax import NeoConfig, run_neo
from neo_jax import driver
from neo_jax.data_models import BoozerData


REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "constellaration"


@pytest.mark.parametrize("iota", [1.00002, -1.00002])
@pytest.mark.parametrize("nfp, field_periods", [(1, 2), (2, 4), (5, 10)])
def test_rational_work_counts_field_periods(iota: float, nfp: int, field_periods: int) -> None:
    work = driver._estimate_rational_work(
        iota, .5, nfp=nfp, nstep_per=3, npart=4, multra=2,
    )
    assert work == dict(abs_iota=abs(iota), field_periods=field_periods,
                        substeps=3*field_periods, eta_paths=24*field_periods)


@pytest.mark.parametrize("iota", [1.00002, -1.00002])
@pytest.mark.parametrize("use_jax, scan", [(False, False), (True, False), (True, True)])
def test_nfp_work_guard_stops_before_integration(
    monkeypatch: pytest.MonkeyPatch, iota: float, use_jax: bool, scan: bool,
) -> None:
    booz = BoozerData(
        rmnc=np.array([[10., 1.]]), zmns=np.array([[0., 1.]]),
        lmns=np.zeros((1, 2)), bmnc=np.array([[1., .1]]),
        ixm=np.array([0, 1]), ixn=np.array([0, 0]), nfp=2,
        es=np.array([.5]), iota=np.array([iota]),
        curr_pol=np.array([10.]), curr_tor=np.zeros(1),
    )
    # These tests exercise the public preflight, without running a transport solve.
    monkeypatch.setattr(driver, "init_surface", lambda *args, **kwargs:
                        SimpleNamespace(bmref=1., b_min=.9, b_max=1.1, splines=None))
    monkeypatch.setenv("NEO_JAX_DISABLE_JIT", "1")
    def unexpected_integration(*args, **kwargs):
        pytest.fail("NFP-aware work guard must run before field-line integration")
    for name in ("flint_bo", "flint_bo_jax", "_solve_surfaces"):
        monkeypatch.setattr(driver, name, unexpected_integration)
    config = NeoConfig(theta_n=8, phi_n=8, npart=3, multra=1, nstep_per=2,
                       nstep_min=2, nstep_max=5, no_bins=2, acc_req=.5,
                       max_rational_field_periods=3)
    with pytest.raises(RuntimeError, match="estimated rational-surface correction is too large"):
        run_neo(booz, config=config, use_jax=use_jax, jax_surface_scan=scan)


@pytest.mark.parametrize(
    "filename",
    [
        "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
        "boozmn_constellaration_DEzsmJ3C22c4ez2YJU24AYU.nc",
    ],
)
def test_constellaration_cases_fail_fast_instead_of_hanging(filename: str) -> None:
    config = NeoConfig(
        surfaces=[1.0e-4],
        theta_n=64,
        phi_n=64,
        npart=40,
    )
    with pytest.raises(RuntimeError, match="estimated rational-surface correction is too large"):
        run_neo(FIXTURE_DIR / filename, config=config, use_jax=True)


def test_constellaration_progress_reports_preflight_diagnostics(capsys: pytest.CaptureFixture[str]) -> None:
    config = NeoConfig(
        surfaces=[1.0e-4],
        theta_n=64,
        phi_n=64,
        npart=40,
    )
    with pytest.raises(RuntimeError, match="max_rational_field_periods=100000"):
        run_neo(
            FIXTURE_DIR / "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
            config=config,
            use_jax=True,
            progress=True,
        )

    out = capsys.readouterr().out
    assert "sqrt(s)=" in out
    assert "approx_rational_field_periods=" in out
    assert "approx_eta_paths=" in out


def test_constellaration_guard_applies_to_jax_surface_scan() -> None:
    config = NeoConfig(
        surfaces=[1.0e-4],
        theta_n=64,
        phi_n=64,
        npart=40,
    )
    with pytest.raises(RuntimeError, match="estimated rational-surface correction is too large"):
        run_neo(
            FIXTURE_DIR / "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
            config=config,
            use_jax=True,
            jax_surface_scan=True,
        )


def test_constellaration_approximate_mode_returns_controlled_result() -> None:
    config = NeoConfig(
        surfaces=[1.0e-4],
        theta_n=8,
        phi_n=8,
        npart=6,
        multra=1,
        nstep_per=4,
        nstep_min=20,
        nstep_max=40,
        no_bins=8,
        acc_req=0.05,
        max_rational_field_periods=12_000,
        rational_surface_policy="approximate",
    )
    results = run_neo(
        FIXTURE_DIR / "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
        config=config,
        use_jax=True,
    )

    assert np.isfinite(results.epsilon_effective[0])
    assert results[0].diagnostics["approximation_used"] is True
    assert "skipping the expensive rational-surface correction" in results[0].diagnostics["approximation_note"]


def test_constellaration_approximate_mode_reports_exact_knob(capsys: pytest.CaptureFixture[str]) -> None:
    config = NeoConfig(
        surfaces=[1.0e-4],
        theta_n=8,
        phi_n=8,
        npart=6,
        multra=1,
        nstep_per=4,
        nstep_min=20,
        nstep_max=40,
        no_bins=8,
        acc_req=0.05,
        max_rational_field_periods=12_000,
        rational_surface_policy="approximate",
    )
    run_neo(
        FIXTURE_DIR / "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
        config=config,
        use_jax=True,
        progress=True,
    )

    out = capsys.readouterr().out
    assert "approximate rational correction enabled" in out
    assert "max_rational_field_periods=0" in out


def test_constellaration_approximate_mode_matches_between_backends() -> None:
    config = NeoConfig(
        surfaces=[1.0e-4],
        theta_n=8,
        phi_n=8,
        npart=6,
        multra=1,
        nstep_per=4,
        nstep_min=20,
        nstep_max=40,
        no_bins=8,
        acc_req=0.05,
        max_rational_field_periods=12_000,
        rational_surface_policy="approximate",
    )
    jax_res = run_neo(
        FIXTURE_DIR / "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
        config=config,
        use_jax=True,
    )
    py_res = run_neo(
        FIXTURE_DIR / "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
        config=config,
        use_jax=False,
    )

    assert jax_res[0].diagnostics["approximation_used"] is True
    assert py_res[0].diagnostics["approximation_used"] is True
    assert np.allclose(jax_res.epsilon_effective, py_res.epsilon_effective, rtol=1e-3, atol=1e-6)


def test_constellaration_approximate_mode_falls_back_from_jax_surface_scan() -> None:
    config = NeoConfig(
        surfaces=[1.0e-4],
        theta_n=8,
        phi_n=8,
        npart=6,
        multra=1,
        nstep_per=4,
        nstep_min=20,
        nstep_max=40,
        no_bins=8,
        acc_req=0.05,
        max_rational_field_periods=12_000,
        rational_surface_policy="approximate",
    )
    results = run_neo(
        FIXTURE_DIR / "boozmn_constellaration_DAEZXsnvNQp3dpxbrdiUW7t.nc",
        config=config,
        use_jax=True,
        jax_surface_scan=True,
    )
    assert np.isfinite(results.epsilon_effective[0])
    assert results[0].diagnostics["rational_surface_policy"] == "approximate"
    assert results[0].diagnostics["approximation_used"] is True
