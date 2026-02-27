from pathlib import Path

import numpy as np

from neo_jax import NeoConfig
from neo_jax.api import load_boozmn, run_booz_xform, run_boozer, run_boozmn
from neo_jax.results import NeoResults


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _orbits_fast_paths():
    boozmn = FIXTURES / "orbits" / "boozmn_ORBITS_FAST.nc"
    return boozmn


def test_run_boozmn_basic():
    boozmn = _orbits_fast_paths()
    config = NeoConfig(surfaces=[64, 96], theta_n=25, phi_n=25)
    results = run_boozmn(boozmn, config=config, use_jax=True)

    assert isinstance(results, NeoResults)
    assert len(results) == 2
    assert results.epsilon_effective.shape == (2,)
    assert results[0].epsilon_effective_by_class.ndim == 1


def test_run_boozer_matches_boozmn():
    boozmn = _orbits_fast_paths()
    config = NeoConfig(surfaces=[64, 96], theta_n=25, phi_n=25)

    booz = load_boozmn(boozmn, surfaces=config.surfaces)
    res_boozmn = run_boozmn(boozmn, config=config, use_jax=True)
    res_boozer = run_boozer(booz, config=config, use_jax=True)

    assert np.allclose(res_boozer.epsilon_effective, res_boozmn.epsilon_effective, rtol=1e-6, atol=1e-10)


def test_run_booz_xform_dict():
    boozmn = _orbits_fast_paths()
    # booz_xform-style data uses 1..ns indexing; use packed surfaces (1,2).
    config = NeoConfig(surfaces=[1, 2], theta_n=25, phi_n=25)

    import netCDF4

    with netCDF4.Dataset(boozmn) as ds:
        booz = {
            "nfp_b": ds.variables["nfp_b"][:],
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

    res_booz_xform = run_booz_xform(booz, config=config, use_jax=True)

    assert len(res_booz_xform) == 2
    assert res_booz_xform.epsilon_effective.shape == (2,)


def test_results_alias_access():
    boozmn = _orbits_fast_paths()
    config = NeoConfig(surfaces=[64], theta_n=25, phi_n=25)
    results = run_boozmn(boozmn, config=config, use_jax=True)

    assert results[0]["epstot"] == results[0].epsilon_effective
    assert np.isclose(results["epstot"][0], results.epsilon_effective[0])
