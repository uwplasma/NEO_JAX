"""Autodiff example: optimize a scalar (rt0) to hit a target epstot."""

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from neo_jax.control import read_control
from neo_jax.driver import compute_reference
from neo_jax.grids import prepare_grids
from neo_jax.integrate import FlintParams, RhsEnv, flint_bo_jax
from neo_jax.io import read_boozmn
from neo_jax.surface import init_surface


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    control_path = repo_root / "tests" / "fixtures" / "ncsx" / "neo_in.ncsx_c09r00_free"
    boozmn_path = repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"

    control = read_control(control_path)
    control = replace(control, fluxs_arr=[9])

    booz = read_boozmn(str(boozmn_path), max_m_mode=control.max_m_mode, max_n_mode=control.max_n_mode)
    grid = prepare_grids(control.theta_n, control.phi_n, booz.nfp)

    surf_idx = control.fluxs_arr[0] - 1
    coeffs = {
        "rmnc": jnp.asarray(booz.rmnc[surf_idx]),
        "zmns": jnp.asarray(booz.zmns[surf_idx]),
        "lmns": jnp.asarray(booz.lmns[surf_idx]),
        "bmnc": jnp.asarray(booz.bmnc[surf_idx]),
    }

    surface = init_surface(
        grid["theta_arr"],
        grid["phi_arr"],
        coeffs,
        jnp.asarray(booz.ixm),
        jnp.asarray(booz.ixn),
        nfp=booz.nfp,
        max_m_mode=control.max_m_mode,
        max_n_mode=control.max_n_mode,
        curr_pol=jnp.asarray(booz.curr_pol[surf_idx]),
        curr_tor=jnp.asarray(booz.curr_tor[surf_idx]),
        iota=jnp.asarray(booz.iota[surf_idx]),
        grid=grid,
    )

    env = RhsEnv(
        splines=surface.splines,
        grid=grid,
        eta=jnp.array([0.0]),
        bmod0=surface.bmref,
        iota=jnp.asarray(booz.iota[surf_idx]),
        curr_pol=jnp.asarray(booz.curr_pol[surf_idx]),
        curr_tor=jnp.asarray(booz.curr_tor[surf_idx]),
    )

    params = FlintParams(
        npart=control.npart,
        multra=control.multra,
        nstep_per=control.nstep_per,
        nstep_min=control.nstep_min,
        nstep_max=control.nstep_max,
        acc_req=control.acc_req,
        no_bins=control.no_bins,
        calc_nstep_max=control.calc_nstep_max,
    )

    ref = compute_reference(booz)
    rt0_base = jnp.asarray(ref["rt0"])

    def epstot_from_rt0(rt0: jnp.ndarray) -> jnp.ndarray:
        out = flint_bo_jax(surface, params, env, nfp=booz.nfp, rt0=rt0)
        return out["epstot"]

    epstot_fn = jax.jit(epstot_from_rt0, static_argnames=())

    base_epstot = epstot_fn(rt0_base)
    target = 0.9 * base_epstot

    def loss(rt0: jnp.ndarray) -> jnp.ndarray:
        epstot = epstot_fn(rt0)
        return (epstot - target) ** 2

    grad_loss = jax.grad(loss)

    rt0 = rt0_base
    lr = 1.0e-2

    for step in range(5):
        loss_val = loss(rt0)
        grad_val = grad_loss(rt0)
        rt0 = rt0 - lr * grad_val
        print(f"step={step} rt0={float(rt0):.6e} loss={float(loss_val):.6e}")

    print("base epstot:", float(base_epstot))
    print("target epstot:", float(target))
    print("final epstot:", float(epstot_fn(rt0)))


if __name__ == "__main__":
    main()
