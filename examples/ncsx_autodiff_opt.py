"""Autodiff optimization example for NEO_JAX."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp

from neo_jax import NeoConfig, load_boozmn, run_boozmn
from neo_jax.driver import compute_reference
from neo_jax.grids import prepare_grids
from neo_jax.integrate import FlintParams, RhsEnv, flint_bo_jax
from neo_jax.surface import init_surface


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    boozmn_path = repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"

    # Baseline run with the high-level API
    config = NeoConfig(surfaces=[19], theta_n=64, phi_n=64)
    baseline = run_boozmn(boozmn_path, config=config, use_jax=True)
    print("baseline epsilon_effective:", baseline.epsilon_effective)

    # Low-level setup for autodiff through a single surface
    booz = load_boozmn(boozmn_path)
    grid = prepare_grids(64, 64, booz.nfp)

    surf_idx = config.surfaces[0] - 1
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
        max_m_mode=int(jnp.max(jnp.abs(booz.ixm))),
        max_n_mode=int(jnp.max(jnp.abs(booz.ixn))),
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
        npart=40,
        multra=2,
        nstep_per=20,
        nstep_min=200,
        nstep_max=500,
        acc_req=0.02,
        no_bins=50,
        calc_nstep_max=0,
    )

    ref = compute_reference(booz)
    rt0_base = jnp.asarray(ref["rt0"])

    def epstot_from_rt0(rt0: jnp.ndarray) -> jnp.ndarray:
        out = flint_bo_jax(surface, params, env, nfp=booz.nfp, rt0=rt0)
        return out["epstot"]

    epstot_fn = jax.jit(epstot_from_rt0)
    base_epstot = epstot_fn(rt0_base)
    target = 0.9 * base_epstot

    def loss(rt0: jnp.ndarray) -> jnp.ndarray:
        return (epstot_fn(rt0) - target) ** 2

    grad_loss = jax.grad(loss)

    rt0 = rt0_base
    lr = 1.0e-2
    for step in range(5):
        loss_val = loss(rt0)
        grad_val = grad_loss(rt0)
        rt0 = rt0 - lr * grad_val
        print(f"step={step} rt0={float(rt0):.6e} loss={float(loss_val):.6e}")

    print("target epstot:", float(target))
    print("final epstot:", float(epstot_fn(rt0)))


if __name__ == "__main__":
    main()
