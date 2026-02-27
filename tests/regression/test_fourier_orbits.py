import pathlib

import numpy as np
import jax.numpy as jnp

from neo_jax.fourier import derived_quantities, fourier_sums
from neo_jax.grids import prepare_grids


FIXTURE_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "orbits"


def _load_vector(name):
    text = (FIXTURE_DIR / name).read_text()
    text = text.replace("D", "E")
    data = np.fromstring(text, sep=" ")
    return data


def _load_matrix(name, shape):
    data = _load_vector(name)
    return data.reshape(shape)


def test_fourier_sums_match_orbits_surface():
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

    # The debug arrays correspond to the last surface in the run.
    s_idx = ns - 1

    fourier = fourier_sums(
        jnp.asarray(theta_arr),
        jnp.asarray(phi_arr),
        jnp.asarray(rmnc[s_idx]),
        jnp.asarray(zmns[s_idx]),
        jnp.asarray(lmns[s_idx]),
        jnp.asarray(bmnc[s_idx]),
        jnp.asarray(ixm),
        jnp.asarray(ixn),
        nfp=int(nfp),
        max_m_mode=int(np.max(np.abs(ixm))),
        max_n_mode=int(np.max(np.abs(ixn))),
    )

    derived = derived_quantities(
        fourier,
        curr_pol=jnp.asarray(curr_pol[s_idx]),
        curr_tor=jnp.asarray(curr_tor[s_idx]),
        iota=jnp.asarray(iota[s_idx]),
    )

    # Compare with fixture outputs
    shape = (theta_n, phi_n)
    b_ref = _load_matrix("b_s_arr.dat", shape)
    r_ref = _load_matrix("r_s_arr.dat", shape)
    z_ref = _load_matrix("z_s_arr.dat", shape)
    l_ref = _load_matrix("l_s_arr.dat", shape)
    sqrg11_ref = _load_matrix("sqrg11_arr.dat", shape)
    kg_ref = _load_matrix("kg_arr.dat", shape)
    pard_ref = _load_matrix("pard_arr.dat", shape)
    p_tb_ref = _load_matrix("p_tb_arr.dat", shape)
    p_pb_ref = _load_matrix("p_pb_arr.dat", shape)

    assert np.allclose(np.asarray(fourier["b"]), b_ref, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.asarray(fourier["r"]), r_ref, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.asarray(fourier["z"]), z_ref, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.asarray(fourier["l"]), l_ref, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.asarray(fourier["p_tb"]), p_tb_ref, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.asarray(fourier["p_pb"]), p_pb_ref, rtol=1e-6, atol=1e-8)

    assert np.allclose(np.asarray(derived["sqrg11"]), sqrg11_ref, rtol=1e-5, atol=1e-8)
    assert np.allclose(np.asarray(derived["kg"]), kg_ref, rtol=1e-6, atol=1e-8)
    assert np.allclose(np.asarray(derived["pard"]), pard_ref, rtol=1e-5, atol=1e-8)


def test_fourier_streamed_matches_vectorized(monkeypatch):
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

    monkeypatch.setenv("NEO_JAX_FOURIER_MODE", "vectorized")
    vec = fourier_sums(
        jnp.asarray(theta_arr),
        jnp.asarray(phi_arr),
        jnp.asarray(rmnc[s_idx]),
        jnp.asarray(zmns[s_idx]),
        jnp.asarray(lmns[s_idx]),
        jnp.asarray(bmnc[s_idx]),
        jnp.asarray(ixm),
        jnp.asarray(ixn),
        nfp=int(nfp),
        max_m_mode=int(np.max(np.abs(ixm))),
        max_n_mode=int(np.max(np.abs(ixn))),
    )
    vec_derived = derived_quantities(
        vec,
        curr_pol=jnp.asarray(curr_pol[s_idx]),
        curr_tor=jnp.asarray(curr_tor[s_idx]),
        iota=jnp.asarray(iota[s_idx]),
    )

    monkeypatch.setenv("NEO_JAX_FOURIER_MODE", "streamed")
    streamed = fourier_sums(
        jnp.asarray(theta_arr),
        jnp.asarray(phi_arr),
        jnp.asarray(rmnc[s_idx]),
        jnp.asarray(zmns[s_idx]),
        jnp.asarray(lmns[s_idx]),
        jnp.asarray(bmnc[s_idx]),
        jnp.asarray(ixm),
        jnp.asarray(ixn),
        nfp=int(nfp),
        max_m_mode=int(np.max(np.abs(ixm))),
        max_n_mode=int(np.max(np.abs(ixn))),
    )
    streamed_derived = derived_quantities(
        streamed,
        curr_pol=jnp.asarray(curr_pol[s_idx]),
        curr_tor=jnp.asarray(curr_tor[s_idx]),
        iota=jnp.asarray(iota[s_idx]),
    )

    for key in ("b", "r", "z", "l", "p_tb", "p_pb"):
        assert np.allclose(np.asarray(vec[key]), np.asarray(streamed[key]), rtol=1e-8, atol=1e-10)

    for key in ("sqrg11", "kg", "pard"):
        assert np.allclose(
            np.asarray(vec_derived[key]), np.asarray(streamed_derived[key]), rtol=1e-8, atol=1e-10
        )
