"""Workflow helpers for building surface problems and scaled outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from .config import NeoConfig
from .data_models import BoozerData
from .driver import compute_reference
from .grids import prepare_grids
from .integrate import FlintParams, RhsEnv
from .surface import init_surface


@dataclass(frozen=True)
class SurfaceProblem:
    surface: object
    env: RhsEnv
    params: FlintParams
    Rmajor: float
    scale: float
    surface_index: int


def resolve_surface_index(booz: BoozerData, surface: int | float) -> int:
    """Resolve a surface selection to a local index (0-based)."""
    if isinstance(surface, float) and 0.0 <= surface <= 1.0:
        s_vals = np.asarray(booz.es)
        best = int(np.argmin(np.abs(s_vals - surface)))
        return best
    idx = int(surface) - 1
    if idx < 0 or idx >= len(booz.es):
        raise ValueError(f"Surface index {surface} is out of range for this BoozerData")
    return idx


def build_surface_problem(
    booz: BoozerData,
    config: NeoConfig,
    *,
    surface: int | float,
) -> SurfaceProblem:
    """Build the surface/env/params bundle for a single surface.

    This is intended for advanced workflows (e.g., autodiff or custom solvers)
    so users do not need to manually wire `init_surface` and `RhsEnv`.
    """
    grid = prepare_grids(config.theta_n, config.phi_n, booz.nfp)
    surf_idx = resolve_surface_index(booz, surface)

    coeffs = {
        "rmnc": jnp.asarray(booz.rmnc[surf_idx]),
        "zmns": jnp.asarray(booz.zmns[surf_idx]),
        "lmns": jnp.asarray(booz.lmns[surf_idx]),
        "bmnc": jnp.asarray(booz.bmnc[surf_idx]),
    }

    max_m_mode = config.max_m_mode if config.max_m_mode > 0 else int(np.max(np.abs(booz.ixm)))
    max_n_mode = config.max_n_mode if config.max_n_mode > 0 else int(np.max(np.abs(booz.ixn)))

    surface_data = init_surface(
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
        splines=surface_data.splines,
        grid=grid,
        eta=jnp.array([0.0]),
        bmod0=surface_data.bmref,
        iota=jnp.asarray(booz.iota[surf_idx]),
        curr_pol=jnp.asarray(booz.curr_pol[surf_idx]),
        curr_tor=jnp.asarray(booz.curr_tor[surf_idx]),
    )

    params = FlintParams(
        npart=config.npart,
        multra=config.multra,
        nstep_per=config.nstep_per,
        nstep_min=config.nstep_min,
        nstep_max=config.nstep_max,
        acc_req=config.acc_req,
        no_bins=config.no_bins,
        calc_nstep_max=config.calc_nstep_max,
    )

    ref = compute_reference(booz)
    Rmajor = float(ref["Rmajor"])

    if config.ref_swi == 1:
        b_ref = ref["bmref_g"]
        r_ref = Rmajor
    elif config.ref_swi == 2:
        b_ref = float(surface_data.bmref)
        r_ref = Rmajor
    else:
        raise ValueError(f"Unsupported ref_swi: {config.ref_swi}")

    scale = (b_ref / float(surface_data.bmref)) ** 2 * (r_ref / Rmajor) ** 2

    return SurfaceProblem(
        surface=surface_data,
        env=env,
        params=params,
        Rmajor=Rmajor,
        scale=scale,
        surface_index=surf_idx,
    )
