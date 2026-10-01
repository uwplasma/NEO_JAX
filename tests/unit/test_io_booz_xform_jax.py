import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax.io import booz_xform_to_boozerdata, booz_xform_to_boozerdata_jax


def test_booz_xform_to_boozerdata_jax_s_override():
    booz = {
        "nfp_b": jnp.array(2),
        "ixm_b": jnp.array([0, 1]),
        "ixn_b": jnp.array([0, 0]),
        "iota_b": jnp.array([0.1, 0.2, 0.3]),
        "buco_b": jnp.array([1.0, 1.1, 1.2]),
        "bvco_b": jnp.array([2.0, 2.1, 2.2]),
        "rmnc_b": jnp.array([[1.0, 0.1], [1.1, 0.2], [1.2, 0.3]]),
        "zmns_b": jnp.array([[0.0, 0.01], [0.0, 0.02], [0.0, 0.03]]),
        "pmns_b": jnp.array([[0.0, 0.05], [0.0, 0.06], [0.0, 0.07]]),
        "bmnc_b": jnp.array([[1.0, 0.2], [1.1, 0.3], [1.2, 0.4]]),
        "s_b": jnp.array([0.2, 0.5, 0.8]),
    }

    data = booz_xform_to_boozerdata_jax(booz, max_m_mode=0, fluxs_arr=[1, 3])
    assert isinstance(data.rmnc, jax.Array)
    assert np.allclose(np.asarray(data.es), [0.2, 0.8])
    np.testing.assert_array_equal(data.iota, [0.1, 0.3])
    np.testing.assert_array_equal(data.curr_tor, [1.0, 1.2])
    np.testing.assert_array_equal(data.curr_pol, [2.0, 2.2])
    assert data.rmnc.shape == (2, 2)


@pytest.mark.parametrize("kind", ["file", "half", "packed", "missing", "short", "outside", "ambiguous"])
@pytest.mark.parametrize("use_jax", [False, True])
def test_profile_packing_and_selection(kind, use_jax):
    booz = dict(nfp_b=2, ixm_b=np.array([0, 1]), ixn_b=np.array([0, 0]),
                rmnc_b=np.arange(6.).reshape(3, 2), zmns_b=np.zeros((3, 2)),
                pmns_b=np.zeros((3, 2)), bmnc_b=np.ones((3, 2)))
    profile = np.arange(6.)/10
    if kind == "half":
        booz.update(compute_surfs=[4, 0, 2], ns_in=5)
        profile = profile[1:]
    else:
        booz.update(jlist=np.array([6, 2, 4]), ns_b=6)
        if kind in ("packed", "ambiguous"):
            profile = profile[[5, 1, 3]]
    booz.update(iota_b=profile, buco_b=profile+1, bvco_b=profile+2)
    if kind in ("missing", "short", "outside", "ambiguous"):
        if kind == "missing":
            del booz["jlist"]
        elif kind == "ambiguous":
            del booz["ns_b"]
        else:
            booz["jlist"] = [6, 2] if kind == "short" else [7, 2, 4]
        with pytest.raises(ValueError, match="profiles|metadata"):
            booz_xform_to_boozerdata(booz, use_jax=use_jax)
        return
    data = booz_xform_to_boozerdata(booz, use_jax=use_jax, fluxs_arr=[3, 1])
    for name, offset in (("iota", 0), ("curr_tor", 1), ("curr_pol", 2)):
        np.testing.assert_allclose(getattr(data, name), np.array([0.3, 0.5])+offset, atol=1e-15)
    np.testing.assert_allclose(data.es, [0.5, 0.9], atol=1e-15)
    np.testing.assert_array_equal(data.rmnc, booz["rmnc_b"][[2, 0]])


def test_full_profiles_keep_derivatives():
    booz = dict(nfp_b=2, ns_b=jnp.array(6), jlist=jnp.array([6, 2, 4]), s_b=jnp.array([.9, .1, .5]),
                ixm_b=np.array([0, 1]), ixn_b=np.array([0, 0]),
                rmnc_b=np.ones((3, 2)), zmns_b=np.zeros((3, 2)),
                pmns_b=np.zeros((3, 2)), bmnc_b=np.ones((3, 2)),
                buco_b=np.arange(6.), bvco_b=np.arange(6.))
    def objective(profile):
        data = booz_xform_to_boozerdata_jax(dict(booz, iota_b=profile),
                                           nfp_override=2, mode_indices=[0, 1], fluxs_arr=[3, 1])
        return jnp.dot(data.iota, jnp.array([2., 3.]))
    np.testing.assert_array_equal(jax.jit(jax.grad(objective))(jnp.arange(6.)), [0, 0, 0, 2, 0, 3])
