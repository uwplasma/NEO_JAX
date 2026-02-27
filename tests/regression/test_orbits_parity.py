import os
from pathlib import Path

import numpy as np
import pytest

from neo_jax.control import read_control
from neo_jax.driver import run_neo_from_boozmn


def _load_fixture(fixtures: Path, *, full: bool):
    if full:
        control = read_control(fixtures / "neo_in.ORBITS")
        boozmn = fixtures / "boozmn_ORBITS.nc"
        ref = np.loadtxt(fixtures / "neo_out.ORBITS")
    else:
        control = read_control(fixtures / "neo_in.ORBITS_FAST")
        boozmn = fixtures / "boozmn_ORBITS_FAST.nc"
        ref = np.loadtxt(fixtures / "neo_out.ORBITS_FAST")
    return control, boozmn, ref


def _assert_results(results, ref):
    assert ref.shape[0] == len(results)

    flux_idx = ref[:, 0].astype(int)
    epstot_ref = ref[:, 1]
    reff_ref = ref[:, 2]
    iota_ref = ref[:, 3]
    b_ref_ref = ref[:, 4]
    r_ref_ref = ref[:, 5]

    out_flux = np.array([r["flux_index"] for r in results], dtype=int)
    epstot = np.array([r["epstot"] for r in results], dtype=float)
    reff = np.array([r["reff"] for r in results], dtype=float)
    iota = np.array([r["iota"] for r in results], dtype=float)
    b_ref = np.array([r["b_ref"] for r in results], dtype=float)
    r_ref = np.array([r["r_ref"] for r in results], dtype=float)

    assert np.array_equal(out_flux, flux_idx)
    assert np.allclose(epstot, epstot_ref, rtol=1e-6, atol=1e-10)
    assert np.allclose(reff, reff_ref, rtol=1e-6, atol=1e-10)
    assert np.allclose(iota, iota_ref, rtol=1e-6, atol=1e-10)
    assert np.allclose(b_ref, b_ref_ref, rtol=1e-6, atol=1e-10)
    assert np.allclose(r_ref, r_ref_ref, rtol=1e-6, atol=1e-10)


def test_orbits_parity_fast():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "orbits"
    control, boozmn, ref = _load_fixture(fixtures, full=False)

    results = run_neo_from_boozmn(str(boozmn), control, use_jax=True)
    _assert_results(results, ref)


def test_orbits_parity_full():
    if not os.getenv("NEO_JAX_ORBITS_FULL"):
        pytest.skip("Set NEO_JAX_ORBITS_FULL=1 to run full ORBITS parity.")

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "orbits"
    control, boozmn, ref = _load_fixture(fixtures, full=True)

    results = run_neo_from_boozmn(str(boozmn), control, use_jax=True)
    _assert_results(results, ref)
