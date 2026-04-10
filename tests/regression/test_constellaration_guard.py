"""Regression coverage for low-iota constellaration cases that used to appear to hang."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from neo_jax import NeoConfig, run_neo


REPO = Path(__file__).resolve().parents[2]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "constellaration"


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
