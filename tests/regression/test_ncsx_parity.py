import os
from pathlib import Path

import numpy as np
import pytest

from neo_jax.control import read_control
from neo_jax.driver import run_neo_from_boozmn


def _assert_parity(
    results,
    ref,
    *,
    epstot_rtol: float = 1e-3,
    reff_rtol: float = 1e-6,
):
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
    # NCSX is sensitive to trapped-orbit accumulation; keep tolerances tight
    # enough to catch regressions while allowing remaining rounding differences.
    assert np.allclose(epstot, epstot_ref, rtol=epstot_rtol, atol=1e-9)
    assert np.allclose(reff, reff_ref, rtol=reff_rtol, atol=1e-9)
    assert np.allclose(iota, iota_ref, rtol=1e-6, atol=1e-9)
    assert np.allclose(b_ref, b_ref_ref, rtol=1e-6, atol=1e-9)
    assert np.allclose(r_ref, r_ref_ref, rtol=1e-6, atol=1e-9)


def test_ncsx_parity_fast():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "ncsx"
    control = read_control(fixtures / "neo_in.ncsx_c09r00_free_fast")
    boozmn = fixtures / "boozmn_ncsx_c09r00_free.nc"

    results = run_neo_from_boozmn(str(boozmn), control, use_jax=True)
    ref = np.loadtxt(fixtures / "neo_out.ncsx_c09r00_free_fast")

    _assert_parity(results, ref, epstot_rtol=6e-3, reff_rtol=1e-5)


def test_ncsx_parity_full():
    if not os.getenv("NEO_JAX_RUN_SLOW"):
        pytest.skip("Set NEO_JAX_RUN_SLOW=1 to run full NCSX parity.")

    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "ncsx"
    control = read_control(fixtures / "neo_in.ncsx_c09r00_free")
    boozmn = fixtures / "boozmn_ncsx_c09r00_free.nc"

    results = run_neo_from_boozmn(str(boozmn), control, use_jax=True)
    ref = np.loadtxt(fixtures / "neo_out.ncsx_c09r00_free")

    _assert_parity(results, ref, epstot_rtol=1e-3)
