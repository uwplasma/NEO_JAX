from pathlib import Path

import numpy as np

from neo_jax.control import read_control
from neo_jax.driver import run_neo_from_boozmn


def test_orbits_parity_jax():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "orbits"
    control = read_control(fixtures / "neo_in.ORBITS")
    boozmn = fixtures / "boozmn_ORBITS.nc"

    results = run_neo_from_boozmn(str(boozmn), control, use_jax=True)

    ref = np.loadtxt(fixtures / "neo_out.ORBITS")
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
