import numpy as np
import jax.numpy as jnp

from neo_jax.integrate import NPQ, RhsEnv, RhsState, rhs_bo1
from neo_jax.splines import spl2d


def test_rhs_bo1_constant_field():
    theta_n, phi_n = 8, 9
    theta_arr = jnp.linspace(0.0, 2.0 * np.pi, theta_n)
    phi_arr = jnp.linspace(0.0, 2.0 * np.pi, phi_n)

    b = jnp.ones((theta_n, phi_n)) * 2.0
    gval = jnp.ones((theta_n, phi_n)) * 3.0
    kg = jnp.ones((theta_n, phi_n)) * 4.0
    pard = jnp.ones((theta_n, phi_n)) * 0.1

    theta_int = float(theta_arr[1] - theta_arr[0])
    phi_int = float(phi_arr[1] - phi_arr[0])

    splines = {
        "b_spl": spl2d(b, theta_int, phi_int, 1, 1),
        "g_spl": spl2d(gval, theta_int, phi_int, 1, 1),
        "k_spl": spl2d(kg, theta_int, phi_int, 1, 1),
        "p_spl": spl2d(pard, theta_int, phi_int, 1, 1),
    }

    grid = {
        "theta_int": theta_int,
        "phi_int": phi_int,
        "mt": 1,
        "mp": 1,
        "theta_start": float(theta_arr[0]),
        "theta_end": float(theta_arr[-1]),
        "phi_start": float(phi_arr[0]),
        "phi_end": float(phi_arr[-1]),
    }

    eta = jnp.array([0.5, 1.0, 2.0])
    bmod0 = jnp.array(2.0)
    iota = jnp.array(0.7)

    env = RhsEnv(splines=splines, grid=grid, eta=eta, bmod0=bmod0, iota=iota)

    npart = eta.shape[0]
    y = jnp.zeros(NPQ + 2 * npart)
    y = y.at[0].set(0.3)

    state = RhsState(
        isw=jnp.zeros(npart, dtype=jnp.int32),
        ipa=jnp.zeros(npart, dtype=jnp.int32),
        icount=jnp.zeros(npart, dtype=jnp.int32),
        ipmax=jnp.array(0, dtype=jnp.int32),
        pard0=jnp.array(0.0),
    )

    dery, new_state = rhs_bo1(jnp.array(0.1), y, state, env)

    # Check main derivatives
    assert np.isclose(np.asarray(dery[0]), 0.7)
    assert np.isclose(np.asarray(dery[1]), 0.25)
    assert np.isclose(np.asarray(dery[2]), 0.75)
    assert np.isclose(np.asarray(dery[3]), 0.5)

    # Check particle derivatives
    p_i = np.asarray(dery[NPQ : NPQ + npart])
    p_h = np.asarray(dery[NPQ + npart : NPQ + 2 * npart])

    # For eta=0.5, subsq < 0 => p_i, p_h = 0
    assert np.isclose(p_i[0], 0.0)
    assert np.isclose(p_h[0], 0.0)

    # For eta=1.0, subsq = 0 => p_i, p_h = 0
    assert np.isclose(p_i[1], 0.0)
    assert np.isclose(p_h[1], 0.0)

    # For eta=2.0, subsq = 0.5 => p_i > 0
    assert p_i[2] > 0.0
    assert p_h[2] != 0.0

    # State updates
    assert int(new_state.ipmax) == 0
    assert np.all(np.asarray(new_state.icount) == np.array([0, 0, 1]))
    assert np.all(np.asarray(new_state.ipa) == np.array([0, 0, 1]))
    assert np.all(np.asarray(new_state.isw) == np.array([0, 0, 1]))
