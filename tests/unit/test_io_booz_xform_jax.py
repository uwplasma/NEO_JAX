import numpy as np
import pytest

import jax
import jax.numpy as jnp

from neo_jax.io import booz_xform_to_boozerdata_jax


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
    assert data.rmnc.shape == (2, 2)
