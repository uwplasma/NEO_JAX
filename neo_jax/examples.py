"""Example workflows for NEO_JAX."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from .api import load_boozmn, run_boozmn
from .config import NeoConfig
from .driver import compute_reference
from .grids import prepare_grids
from .integrate import FlintParams, RhsEnv, flint_bo_jax
from .plotting import plot_epsilon_effective
from .surface import init_surface


def _default_ncsx_boozmn_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"


def ncsx_jit_demo(
    *,
    boozmn_path: str | Path | None = None,
    surfaces: Sequence[int] | None = None,
    theta_n: int = 64,
    phi_n: int = 64,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Run a JIT-compiled NCSX demo and plot epsilon effective."""
    if boozmn_path is None:
        boozmn_path = _default_ncsx_boozmn_path()

    if surfaces is None:
        surfaces = [19, 39, 59, 79]

    config = NeoConfig(
        surfaces=list(surfaces),
        theta_n=theta_n,
        phi_n=phi_n,
    )

    results = run_boozmn(boozmn_path, config=config, use_jax=True, progress=True)

    fig, _ax = plot_epsilon_effective(results, label="NEO_JAX")
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def ncsx_autodiff_demo(
    *,
    boozmn_path: str | Path | None = None,
    surface_index: int = 19,
    steps: int = 5,
    lr: float = 1.0e-2,
) -> None:
    """Autodiff demo: optimize rt0 to hit a target epsilon effective."""
    if boozmn_path is None:
        boozmn_path = _default_ncsx_boozmn_path()

    booz = load_boozmn(boozmn_path)
    grid = prepare_grids(64, 64, booz.nfp)

    surf_idx = surface_index - 1
    coeffs = {
        "rmnc": jnp.asarray(booz.rmnc[surf_idx]),
        "zmns": jnp.asarray(booz.zmns[surf_idx]),
        "lmns": jnp.asarray(booz.lmns[surf_idx]),
        "bmnc": jnp.asarray(booz.bmnc[surf_idx]),
    }

    max_m_mode = int(np.max(np.abs(booz.ixm)))
    max_n_mode = int(np.max(np.abs(booz.ixn)))

    surface = init_surface(
        grid["theta_arr"],
        grid["phi_arr"],
        coeffs,
        jnp.asarray(booz.ixm),
        jnp.asarray(booz.ixn),
        nfp=booz.nfp,
        max_m_mode=max_m_mode,
        max_n_mode=max_n_mode,
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
        epstot = epstot_fn(rt0)
        return (epstot - target) ** 2

    grad_loss = jax.grad(loss)

    rt0 = rt0_base
    for step in range(steps):
        loss_val = loss(rt0)
        grad_val = grad_loss(rt0)
        rt0 = rt0 - lr * grad_val
        print(f"step={step} rt0={float(rt0):.6e} loss={float(loss_val):.6e}")

    print("base epstot:", float(base_epstot))
    print("target epstot:", float(target))
    print("final epstot:", float(epstot_fn(rt0)))
