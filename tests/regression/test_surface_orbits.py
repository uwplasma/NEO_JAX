import pathlib

import numpy as np
import jax.numpy as jnp

from neo_jax.surface import init_surface

FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "orbits"


def _load_vector(name):
    text = (FIXTURE_DIR / name).read_text()
    text = text.replace("D", "E")
    return np.fromstring(text, sep=" ")


def _load_matrix(name, shape):
    return _load_vector(name).reshape(shape)


def test_surface_bmin_bmax_orbits():
    dims = _load_vector("dimension.dat").astype(int)
    ns, mnmax, nfp, theta_n, phi_n, _ = dims

    theta_arr = _load_vector("theta_arr.dat")
    phi_arr = _load_vector("phi_arr.dat")

    mn = _load_matrix("mn_arr.dat", (mnmax, 2)).astype(int)
    ixm = mn[:, 0]
    ixn = mn[:, 1]

    rmnc = _load_vector("rmnc_arr.dat").reshape((ns, mnmax))
    zmns = _load_vector("zmns_arr.dat").reshape((ns, mnmax))
    lmns = _load_vector("lmns_arr.dat").reshape((ns, mnmax))
    bmnc = _load_vector("bmnc_arr.dat").reshape((ns, mnmax))

    es = _load_matrix("es_arr.dat", (ns, 6))
    iota = es[:, 1]
    curr_pol = es[:, 2]
    curr_tor = es[:, 3]

    s_idx = ns - 1

    grid = {
        "theta_int": float(theta_arr[1] - theta_arr[0]),
        "phi_int": float(phi_arr[1] - phi_arr[0]),
        "theta_start": float(theta_arr[0]),
        "theta_end": float(theta_arr[-1]),
        "phi_start": float(phi_arr[0]),
        "phi_end": float(phi_arr[-1]),
        "mt": 1,
        "mp": 1,
    }

    coeffs = {
        "rmnc": jnp.asarray(rmnc[s_idx]),
        "zmns": jnp.asarray(zmns[s_idx]),
        "lmns": jnp.asarray(lmns[s_idx]),
        "bmnc": jnp.asarray(bmnc[s_idx]),
    }

    surface = init_surface(
        jnp.asarray(theta_arr),
        jnp.asarray(phi_arr),
        coeffs,
        jnp.asarray(ixm),
        jnp.asarray(ixn),
        nfp=int(nfp),
        max_m_mode=int(np.max(np.abs(ixm))),
        max_n_mode=int(np.max(np.abs(ixn))),
        curr_pol=jnp.asarray(curr_pol[s_idx]),
        curr_tor=jnp.asarray(curr_tor[s_idx]),
        iota=jnp.asarray(iota[s_idx]),
        grid=grid,
    )

    # Parse diagnostic_add for reference b_min/b_max/bmref
    diag = _load_vector("diagnostic_add.dat")
    # diag format: 4 ints then 6 floats
    b_min_ref = diag[4]
    b_max_ref = diag[5]
    bmref_ref = diag[6]

    assert np.isclose(np.asarray(surface.b_min), b_min_ref, rtol=1e-6, atol=1e-8)
    assert np.isclose(np.asarray(surface.b_max), b_max_ref, rtol=1e-6, atol=1e-8)
    assert np.isclose(np.asarray(surface.bmref), bmref_ref, rtol=1e-6, atol=1e-8)
