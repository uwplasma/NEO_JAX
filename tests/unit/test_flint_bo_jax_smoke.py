from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax.integrate import FlintParams, RhsEnv, flint_bo, flint_bo_jax
from neo_jax.surface import SurfaceData, build_splines


def _constant_case(iota=0.5):
    jax.config.update("jax_enable_x64", True)

    theta_n, phi_n = 6, 6
    theta_arr = jnp.linspace(0.0, 2.0 * np.pi, theta_n)
    phi_arr = jnp.linspace(0.0, 2.0 * np.pi, phi_n)

    b = jnp.ones((theta_n, phi_n)) * 2.0
    sqrg11 = jnp.ones((theta_n, phi_n)) * 3.0
    kg = jnp.ones((theta_n, phi_n)) * 4.0
    pard = jnp.ones((theta_n, phi_n)) * 0.1

    fields = {"b": b, "sqrg11": sqrg11, "kg": kg, "pard": pard}

    theta_int = float(theta_arr[1] - theta_arr[0])
    phi_int = float(phi_arr[1] - phi_arr[0])

    splines = build_splines(fields, theta_int, phi_int, mt=1, mp=1, calc_cur=False)

    surface = SurfaceData(
        b_min=jnp.array(2.0),
        b_max=jnp.array(2.0),
        theta_bmin=jnp.array(0.0),
        phi_bmin=jnp.array(0.0),
        theta_bmax=jnp.array(0.0),
        phi_bmax=jnp.array(0.0),
        bmref=jnp.array(2.0),
        fields=fields,
        splines=splines,
    )

    grid = {
        "theta_int": theta_int,
        "phi_int": phi_int,
        "theta_start": float(theta_arr[0]),
        "theta_end": float(theta_arr[-1]),
        "phi_start": float(phi_arr[0]),
        "phi_end": float(phi_arr[-1]),
        "mt": 1,
        "mp": 1,
    }

    params = FlintParams(
        npart=3,
        multra=1,
        nstep_per=1,
        nstep_min=0,
        nstep_max=2,
        acc_req=0.1,
        no_bins=4,
        calc_nstep_max=0,
    )

    env = RhsEnv(
        splines=surface.splines,
        grid=grid,
        eta=jnp.array([0.0]),
        bmod0=surface.bmref,
        iota=jnp.array(iota),
        curr_pol=jnp.array(1.0),
        curr_tor=jnp.array(0.0),
    )

    return surface, params, env


def test_flint_bo_jax_smoke_constant_field():
    surface, params, env = _constant_case()
    out = flint_bo_jax(surface, params, env, nfp=1, Rmajor=1.0)

    assert np.isfinite(np.asarray(out["epstot"]))
    assert int(out["hit_rat"]) in (0, 1)


@pytest.mark.parametrize("integrator", [flint_bo, flint_bo_jax])
@pytest.mark.parametrize("iota", [0.50001, -0.50001])
def test_rational_period_count_is_positive(integrator, iota):
    surface, params, env = _constant_case(iota)
    params = replace(params, nstep_min=2, nstep_max=5, acc_req=0.5, no_bins=2)
    out = integrator(surface, params, env, nfp=1, Rmajor=1.0)

    assert int(out["hit_rat"]) == 1
    assert int(out["nfp_rat"]) == 4
    assert int(out["final_n"]) == 8
    assert np.isfinite(np.asarray(out["epstot"]))
