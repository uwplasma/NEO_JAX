"""High-level driver for NEO_JAX using Boozer data."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import jax.numpy as jnp

from .control import ControlParams
from .data_models import BoozerData
from .grids import prepare_grids
from .integrate import FlintParams, RhsEnv, flint_bo, flint_bo_jax
from .io import read_boozmn
from .surface import init_surface


def compute_reference(booz: BoozerData) -> Dict[str, float]:
    m0_idx = np.where((booz.ixm == 0) & (booz.ixn == 0))[0][0]
    rt0 = float(booz.rmnc[0, m0_idx])
    bmref_g = float(booz.bmnc[0, m0_idx])
    return {"rt0": rt0, "bmref_g": bmref_g}


def run_neo_from_boozer(
    booz: BoozerData,
    control: ControlParams,
    *,
    use_jax: bool = False,
    progress: bool = False,
) -> List[Dict[str, float]]:
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
        if progress:
            print(f"NEO_JAX: surface {local_idx + 1}/{len(surf_indices)} (index {surf_idx + 1})")
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

        if use_jax:
            out = flint_bo_jax(surface, params, env, nfp=booz.nfp, rt0=rt0)
        else:
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

        result = (
            {
                "flux_index": control.fluxs_arr[local_idx] if control.fluxs_arr else surf_idx + 1,
                "epstot": epstot,
                "reff": reff,
                "iota": float(booz.iota[surf_idx]),
                "b_ref": b_ref,
                "r_ref": r_ref,
                "epspar": epspar,
                "ctrone": float(out.get("ctrone", 0.0)),
                "ctrtot": float(out.get("ctrtot", 0.0)),
                "bareph": float(out.get("bareph", 0.0)),
                "barept": float(out.get("barept", 0.0)),
                "yps": float(out.get("yps", 0.0)),
                "diagnostics": out,
            }
        )
        results.append(result)
        if progress:
            print(
                f"NEO_JAX: epstot={result['epstot']:.6e} reff={result['reff']:.6e} iota={result['iota']:.6e}"
            )

    return results


def run_neo_from_boozmn(
    boozmn_path: str,
    control: ControlParams,
    *,
    use_jax: bool = False,
    progress: bool = False,
    extension: str | None = None,
) -> List[Dict[str, float]]:
    booz = read_boozmn(
        boozmn_path,
        max_m_mode=control.max_m_mode,
        max_n_mode=control.max_n_mode,
        fluxs_arr=control.fluxs_arr,
        extension=extension,
    )
    return run_neo_from_boozer(booz, control, use_jax=use_jax, progress=progress)
