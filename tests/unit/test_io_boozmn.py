import shutil
from pathlib import Path

import netCDF4
import numpy as np
import pytest

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


def test_read_boozmn_preserves_asymmetric_geometry(tmp_path):
    source = Path(__file__).resolve().parents[1] / "fixtures/orbits/boozmn_ORBITS.nc"
    path = tmp_path / "boozmn.nc"
    shutil.copy2(source, path)
    with netCDF4.Dataset(path, "a") as ds:
        ds.variables["lasym__logical__"][...] = 1
        for target, cosine in (("bmns_b", "bmnc_b"), ("rmns_b", "rmnc_b"),
                               ("zmnc_b", "zmns_b"), ("pmnc_b", "pmns_b")):
            sine = ds.createVariable(target, "f8", ds.variables[cosine].dimensions)
            sine[...] = 0.01 * ds.variables[cosine][...]
    data = read_boozmn(path, fluxs_arr=[96, 64], max_m_mode=3)
    reference = read_boozmn(source, fluxs_arr=[96, 64], max_m_mode=3)
    for sine, cosine in (("bmns", "bmnc"), ("rmns", "rmnc"),
                         ("zmnc", "zmns"), ("lmnc", "lmns")):
        np.testing.assert_allclose(getattr(data, sine), .01 * getattr(reference, cosine))


def test_read_boozmn_requires_complete_asymmetric_geometry(tmp_path):
    source = Path(__file__).resolve().parents[1] / "fixtures/orbits/boozmn_ORBITS.nc"
    path = tmp_path / "boozmn.nc"
    shutil.copy2(source, path)
    with netCDF4.Dataset(path, "a") as ds:
        ds.variables["lasym__logical__"][...] = 1
    with pytest.raises(KeyError, match="rmns_b"):
        read_boozmn(path)
