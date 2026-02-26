"""High-level driver for NEO_JAX using Boozer data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp

from .control import ControlParams
from .grids import prepare_grids
from .integrate import FlintParams, RhsEnv, flint_bo
from .surface import init_surface


@dataclass(frozen=True)
class BoozerData:
    rmnc: np.ndarray
    zmns: np.ndarray
    lmns: np.ndarray
    bmnc: np.ndarray
    ixm: np.ndarray
    ixn: np.ndarray
    es: np.ndarray
    iota: np.ndarray
    curr_pol: np.ndarray
    curr_tor: np.ndarray
    nfp: int


def compute_reference(booz: BoozerData) -> Dict[str, float]:
    m0_idx = np.where((booz.ixm == 0) & (booz.ixn == 0))[0][0]
    rt0 = float(booz.rmnc[0, m0_idx])
    bmref_g = float(booz.bmnc[0, m0_idx])
    return {"rt0": rt0, "bmref_g": bmref_g}


def run_neo_from_boozer(booz: BoozerData, control: ControlParams) -> List[Dict[str, float]]:
    grid = prepare_grids(control.theta_n, control.phi_n, booz.nfp)

    max_m_mode = control.max_m_mode if control.max_m_mode > 0 else int(np.max(np.abs(booz.ixm)))
    max_n_mode = control.max_n_mode if control.max_n_mode > 0 else int(np.max(np.abs(booz.ixn)))

    if control.fluxs_arr:
        surf_indices = [i - 1 for i in control.fluxs_arr]
    else:
        surf_indices = list(range(booz.rmnc.shape[0]))

    ref = compute_reference(booz)
    rt0 = ref["rt0"]
    bmref_g = ref["bmref_g"]

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

    results: List[Dict[str, float]] = []
    reff = 0.0

    for local_idx, surf_idx in enumerate(surf_indices):
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

        out = flint_bo(surface, params, env, nfp=booz.nfp, rt0=rt0)

        if control.ref_swi == 1:
            b_ref = bmref_g
            r_ref = rt0
        elif control.ref_swi == 2:
            b_ref = float(surface.bmref)
            r_ref = rt0
        else:
            raise ValueError(f"Unsupported ref_swi: {control.ref_swi}")

        scale = (b_ref / float(surface.bmref)) ** 2 * (r_ref / rt0) ** 2
        epstot = float(out["epstot"] * scale)
        epspar = np.asarray(out["epspar"]) * scale

        psi = booz.es[surf_idx]
        if local_idx == 0:
            dpsi = psi
        else:
            dpsi = psi - booz.es[surf_indices[local_idx - 1]]
        reff = reff + float(out["drdpsi"] * dpsi)

        results.append(
            {
                "flux_index": control.fluxs_arr[local_idx] if control.fluxs_arr else surf_idx + 1,
                "epstot": epstot,
                "reff": reff,
                "iota": float(booz.iota[surf_idx]),
                "b_ref": b_ref,
                "r_ref": r_ref,
                "epspar": epspar,
            }
        )

    return results
