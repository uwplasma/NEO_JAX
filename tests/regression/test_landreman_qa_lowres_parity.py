from pathlib import Path

import numpy as np

from neo_jax.control import read_control
from neo_jax.driver import run_neo_from_boozmn


def test_landreman_qa_lowres_parity():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "landreman_qa_lowres"
    control = read_control(fixtures / "neo_in.LandremanPaul2021_QA_lowres")
    boozmn = fixtures / "boozmn_LandremanPaul2021_QA_lowres.nc"

    results = run_neo_from_boozmn(str(boozmn), control, use_jax=True)

    ref = np.loadtxt(fixtures / "neo_out.LandremanPaul2021_QA_lowres")
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


# eps_eff of the unchanged spline code on the six surfaces (regression pin). The bound
# rtol 1e-10 is that of the existing check that two schedules of one kernel agree.
EPSTOT_PINNED = [3.024589531175728e-09, 5.540962319388874e-09, 1.3159992149842503e-08,
                 2.0108052403464786e-08, 2.4735269991700135e-08, 2.7907212960905335e-08]
B_REF_PINNED = [1.06258273422786, 1.0761731866424438, 1.087518228731963,
                1.0974615943024835, 1.1064159968985756, 1.114646986179379]


def test_landreman_qa_lowres_pinned():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "landreman_qa_lowres"
    control = read_control(fixtures / "neo_in.LandremanPaul2021_QA_lowres")
    results = run_neo_from_boozmn(
        str(fixtures / "boozmn_LandremanPaul2021_QA_lowres.nc"), control, use_jax=True)
    np.testing.assert_allclose([r["epstot"] for r in results], EPSTOT_PINNED, rtol=1e-10)
    np.testing.assert_allclose([r["b_ref"] for r in results], B_REF_PINNED, rtol=1e-10)
