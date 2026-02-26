from pathlib import Path

import netCDF4
import numpy as np

from neo_jax.io import read_boozmn


def test_read_boozmn_orbits():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "orbits"
    booz_path = fixtures / "boozmn_ORBITS.nc"

    fluxs_arr = [2, 4, 8, 16, 32, 64, 96, 120]
    booz = read_boozmn(booz_path, max_m_mode=0, max_n_mode=0, fluxs_arr=fluxs_arr)

    rmnc_ref = np.loadtxt(fixtures / "rmnc_arr.dat").reshape((8, 196))
    zmns_ref = np.loadtxt(fixtures / "zmns_arr.dat").reshape((8, 196))
    lmns_ref = np.loadtxt(fixtures / "lmns_arr.dat").reshape((8, 196))
    bmnc_ref = np.loadtxt(fixtures / "bmnc_arr.dat").reshape((8, 196))

    with netCDF4.Dataset(booz_path) as ds:
        nfp_ref = int(ds.variables["nfp_b"][:])
    assert booz.nfp == nfp_ref
    assert booz.rmnc.shape == rmnc_ref.shape
    assert np.allclose(booz.rmnc, rmnc_ref)
    assert np.allclose(booz.zmns, zmns_ref)
    assert np.allclose(booz.lmns, lmns_ref)
    assert np.allclose(booz.bmnc, bmnc_ref)
