from pathlib import Path

import numpy as np
import pytest

from neo_jax import NeoConfig, api, build_surface_problem
from neo_jax.api import load_boozmn, run_booz_xform, run_neo
from neo_jax.io import booz_xform_to_boozerdata
from neo_jax.results import NeoResults

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _orbits_fast_paths():
    boozmn = FIXTURES / "orbits" / "boozmn_ORBITS_FAST.nc"
    return boozmn


def test_run_neo_boozmn_basic():
    boozmn = _orbits_fast_paths()
    # Surfaces specified by s in [0,1] should map to nearest jlist indices.
    config = NeoConfig(surfaces=[0.5, 0.75], theta_n=25, phi_n=25)
    results = run_neo(boozmn, config=config, use_jax=True)

    assert isinstance(results, NeoResults)
    assert len(results) == 2
    assert results.epsilon_effective.shape == (2,)
    assert results[0].epsilon_effective_by_class.ndim == 1
    # ORBITS_FAST jlist maps s≈0.5->64 and s≈0.75->96.
    assert results[0].flux_index == 64
    assert results[1].flux_index == 96


def test_run_neo_boozer_matches_boozmn():
    boozmn = _orbits_fast_paths()
    config = NeoConfig(surfaces=[64, 96], theta_n=25, phi_n=25)

    booz = load_boozmn(boozmn, surfaces=config.surfaces)
    res_boozmn = run_neo(boozmn, config=config, use_jax=True)
    res_boozer = run_neo(booz, config=config, use_jax=True)

    assert np.allclose(res_boozer.epsilon_effective, res_boozmn.epsilon_effective, rtol=1e-6, atol=1e-10)


@pytest.mark.parametrize("use_jax", [False, True])
@pytest.mark.parametrize("surfaces", [[2, 1], [0.75, 0.5], [2, 0.75]])
def test_run_booz_xform_dict(monkeypatch, use_jax, surfaces):
    boozmn = _orbits_fast_paths()
    config = NeoConfig(surfaces=surfaces, theta_n=25, phi_n=25)

    import netCDF4

    with netCDF4.Dataset(boozmn) as ds:
        booz = {
            "nfp_b": ds.variables["nfp_b"][:],
            "ns_b": ds.variables["ns_b"][:],
            "ixm_b": ds.variables["ixm_b"][:],
            "ixn_b": ds.variables["ixn_b"][:],
            "iota_b": ds.variables["iota_b"][:],
            "buco_b": ds.variables["buco_b"][:],
            "bvco_b": ds.variables["bvco_b"][:],
            "rmnc_b": ds.variables["rmnc_b"][:],
            "zmns_b": ds.variables["zmns_b"][:],
            "pmns_b": ds.variables["pmns_b"][:],
            "bmnc_b": ds.variables["bmnc_b"][:],
            "jlist": ds.variables["jlist"][:],
        }

    es = (booz["jlist"]-1.5)/(int(booz["ns_b"])-1)
    rows = [int(np.argmin(abs(es-s))) if isinstance(s, float) else s-1 for s in surfaces]
    radial = np.asarray(booz["jlist"])[rows]-1
    monkeypatch.setattr(api, "run_boozer", lambda data, **kwargs: data)
    data = run_booz_xform(booz, config=config, use_jax=use_jax)
    for field, raw in (("iota", "iota_b"), ("curr_tor", "buco_b"), ("curr_pol", "bvco_b")):
        np.testing.assert_array_equal(getattr(data, field), np.asarray(booz[raw])[radial])
    np.testing.assert_allclose(data.es, es[rows], rtol=1e-14)
    np.testing.assert_array_equal(data.bmnc, np.asarray(booz["bmnc_b"])[rows])


def test_booz_xform_to_boozerdata_jax():
    boozmn = _orbits_fast_paths()
    import jax
    import jax.numpy as jnp
    import netCDF4

    def _to_jnp(var):
        arr = var[:]
        if isinstance(arr, np.ma.MaskedArray):
            arr = arr.filled()
        return jnp.asarray(arr)

    with netCDF4.Dataset(boozmn) as ds:
        booz = {
            "nfp_b": _to_jnp(ds.variables["nfp_b"]),
            "ns_b": _to_jnp(ds.variables["ns_b"]),
            "jlist": _to_jnp(ds.variables["jlist"]),
            "ixm_b": _to_jnp(ds.variables["ixm_b"]),
            "ixn_b": _to_jnp(ds.variables["ixn_b"]),
            "iota_b": _to_jnp(ds.variables["iota_b"]),
            "buco_b": _to_jnp(ds.variables["buco_b"]),
            "bvco_b": _to_jnp(ds.variables["bvco_b"]),
            "rmnc_b": _to_jnp(ds.variables["rmnc_b"]),
            "zmns_b": _to_jnp(ds.variables["zmns_b"]),
            "pmns_b": _to_jnp(ds.variables["pmns_b"]),
            "bmnc_b": _to_jnp(ds.variables["bmnc_b"]),
        }

    data = booz_xform_to_boozerdata(booz, use_jax=True)
    assert isinstance(data.rmnc, jax.Array)


def test_results_alias_access():
    boozmn = _orbits_fast_paths()
    config = NeoConfig(surfaces=[64], theta_n=25, phi_n=25)
    results = run_neo(boozmn, config=config, use_jax=True)

    assert results[0]["epstot"] == results[0].epsilon_effective
    assert np.isclose(results["epstot"][0], results.epsilon_effective[0])
    assert np.isclose(results["s"][0], results.s[0])
    assert np.isclose(results["sqrt_s"][0], results.sqrt_s[0])
    assert np.isclose(results["r_eff"][0], results.r_eff[0])


def test_jax_surface_scan_normalizes_error_policy() -> None:
    boozmn = _orbits_fast_paths()
    config = NeoConfig(surfaces=[64], theta_n=25, phi_n=25, rational_surface_policy="ERROR")
    results = run_neo(boozmn, config=config, use_jax=True, jax_surface_scan=True)

    assert not isinstance(results, NeoResults)
    assert np.asarray(results.eps_eff).shape == (1,)


def test_jax_surface_kernel_reuses_compile_with_new_coefficients():
    from dataclasses import replace

    from neo_jax.driver import _solve_surfaces

    booz = load_boozmn(_orbits_fast_paths())
    config = NeoConfig(surfaces=[64], theta_n=17, phi_n=17, npart=8, multra=1,
                       nstep_per=8, nstep_min=4, nstep_max=8, max_m_mode=6, max_n_mode=12)
    first = run_neo(booz, config=config, jax_surface_scan=True)
    np.asarray(first.eps_eff)
    compiled = _solve_surfaces._cache_size()
    scaled = replace(booz, bmnc=booz.bmnc * 1.01)
    second = run_neo(scaled, config=config, jax_surface_scan=True)
    np.asarray(second.eps_eff)
    assert _solve_surfaces._cache_size() == compiled
    np.testing.assert_allclose(second.diagnostics['b_ref'],
                               first.diagnostics['b_ref'] * 1.01, rtol=1e-10)


def test_build_surface_problem_maps_s():
    boozmn = _orbits_fast_paths()
    booz = load_boozmn(boozmn)
    config = NeoConfig(surfaces=[0.5], theta_n=25, phi_n=25)
    problem = build_surface_problem(booz, config, surface=0.5)

    assert 0 <= problem.surface_index < len(booz.es)
    assert problem.Rmajor > 0.0
