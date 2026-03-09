"""High-level driver for NEO_JAX using Boozer data."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .control import ControlParams
from .data_models import BoozerData
from .grids import prepare_grids
from .integrate import FlintParams, RhsEnv, flint_bo, flint_bo_jax
from .io import read_boozmn
from .legacy import LegacyNeoWriter, build_fortran_line
from .results import NeoResults, NeoSurfaceResult
from .surface import init_surface
from .data_models import NeoOutputs


def compute_reference(booz: BoozerData) -> Dict[str, float]:
    m0_idx = np.where((booz.ixm == 0) & (booz.ixn == 0))[0][0]
    rt0 = float(booz.rmnc[0, m0_idx])
    bmref_g = float(booz.bmnc[0, m0_idx])
    return {"rt0": rt0, "Rmajor": rt0, "bmref_g": bmref_g}


def compute_reference_jax(booz: BoozerData):
    """JAX-friendly reference values."""
    m0_mask = (booz.ixm == 0) & (booz.ixn == 0)
    m0_idx = jnp.where(m0_mask, size=1, fill_value=0)[0]
    rt0 = jnp.squeeze(booz.rmnc[0, m0_idx])
    bmref_g = jnp.squeeze(booz.bmnc[0, m0_idx])
    return rt0, bmref_g


def run_neo_from_boozer_jax(
    booz: BoozerData,
    control: ControlParams,
    *,
    skip_fourier_mask: bool = False,
) -> NeoOutputs:
    """JAX surface scan over all requested surfaces (no Python loop)."""
    booz = BoozerData(
        rmnc=jnp.asarray(booz.rmnc),
        zmns=jnp.asarray(booz.zmns),
        lmns=jnp.asarray(booz.lmns),
        bmnc=jnp.asarray(booz.bmnc),
        ixm=jnp.asarray(booz.ixm),
        ixn=jnp.asarray(booz.ixn),
        es=jnp.asarray(booz.es),
        iota=jnp.asarray(booz.iota),
        curr_pol=jnp.asarray(booz.curr_pol),
        curr_tor=jnp.asarray(booz.curr_tor),
        nfp=int(booz.nfp),
    )
    grid = prepare_grids(control.theta_n, control.phi_n, booz.nfp)

    def _max_abs_mode(arr):
        if isinstance(arr, jax.Array):
            return jnp.max(jnp.abs(arr))
        return int(np.max(np.abs(arr)))

    max_m_mode = control.max_m_mode if control.max_m_mode > 0 else _max_abs_mode(booz.ixm)
    max_n_mode = control.max_n_mode if control.max_n_mode > 0 else _max_abs_mode(booz.ixn)

    if control.fluxs_arr:
        if booz.rmnc.shape[0] == len(control.fluxs_arr):
            surf_indices = list(range(booz.rmnc.shape[0]))
        else:
            surf_indices = [i - 1 for i in control.fluxs_arr]
        flux_indices = list(control.fluxs_arr)
    else:
        surf_indices = list(range(booz.rmnc.shape[0]))
        flux_indices = [idx + 1 for idx in surf_indices]

    surf_indices_j = jnp.asarray(surf_indices, dtype=jnp.int32)
    flux_indices_j = jnp.asarray(flux_indices, dtype=jnp.int32)

    rt0, bmref_g = compute_reference_jax(booz)

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

    def _solve_surface(surf_idx):
        coeffs = {
            "rmnc": booz.rmnc[surf_idx],
            "zmns": booz.zmns[surf_idx],
            "lmns": booz.lmns[surf_idx],
            "bmnc": booz.bmnc[surf_idx],
        }

        surface = init_surface(
            grid["theta_arr"],
            grid["phi_arr"],
            coeffs,
            booz.ixm,
            booz.ixn,
            nfp=booz.nfp,
            max_m_mode=max_m_mode,
            max_n_mode=max_n_mode,
            curr_pol=booz.curr_pol[surf_idx],
            curr_tor=booz.curr_tor[surf_idx],
            iota=booz.iota[surf_idx],
            grid=grid,
            use_jax=True,
            skip_mask=skip_fourier_mask,
        )

        env = RhsEnv(
            splines=surface.splines,
            grid=grid,
            eta=jnp.array([0.0]),
            bmod0=surface.bmref,
            iota=booz.iota[surf_idx],
            curr_pol=booz.curr_pol[surf_idx],
            curr_tor=booz.curr_tor[surf_idx],
        )

        out = flint_bo_jax(surface, params, env, nfp=booz.nfp, rt0=rt0)

        if control.ref_swi == 1:
            b_ref = bmref_g
            r_ref = rt0
        elif control.ref_swi == 2:
            b_ref = surface.bmref
            r_ref = rt0
        else:
            raise ValueError(f"Unsupported ref_swi: {control.ref_swi}")

        scale = (b_ref / surface.bmref) ** 2 * (r_ref / rt0) ** 2
        epstot = out["epstot"] * scale
        epspar = out["epspar"] * scale

        return (
            epstot,
            epspar,
            out["ctrone"],
            out["ctrtot"],
            out["bareph"],
            out["barept"],
            out["yps"],
            out["drdpsi"],
            surface.bmref,
            booz.es[surf_idx],
            booz.iota[surf_idx],
            b_ref,
            r_ref,
        )

    (
        epstot,
        epspar,
        ctrone,
        ctrtot,
        bareph,
        barept,
        yps,
        drdpsi,
        bmref,
        s_vals,
        iota_vals,
        b_ref,
        r_ref,
    ) = jax.vmap(_solve_surface)(surf_indices_j)

    dpsi = jnp.concatenate([s_vals[:1], s_vals[1:] - s_vals[:-1]], axis=0)
    r_eff = jnp.cumsum(drdpsi * dpsi)

    diagnostics = {
        "s": s_vals,
        "r_eff": r_eff,
        "iota": iota_vals,
        "b_ref": b_ref,
        "r_ref": r_ref,
        "bareph": bareph,
        "barept": barept,
        "yps": yps,
        "flux_index": flux_indices_j,
    }

    return NeoOutputs(
        eps_eff=epstot,
        eps_par=epspar,
        eps_tot=epstot,
        ctr_one=ctrone,
        ctr_tot=ctrtot,
        diagnostics=diagnostics,
    )


def _env_flag(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _write_diagnostic_files(
    *,
    events: Sequence[Tuple[int, int, int, float]],
    meta: Dict[str, float],
    psi_ind: int,
    path_prefix: str = "",
) -> None:
    diag_path = f"{path_prefix}diagnostic.dat"
    with open(diag_path, "w", encoding="utf-8") as handle:
        for i_idx, icount, ipa, add_on in events:
            handle.write(
                build_fortran_line(
                    (int(i_idx), int(icount), int(ipa)),
                    int_width=8,
                    reals=(float(add_on),),
                    real_width=20,
                    real_digits=10,
                    real_letter="E",
                )
                + "\n"
            )

    add_path = f"{path_prefix}diagnostic_add.dat"
    with open(add_path, "w", encoding="utf-8") as handle:
        handle.write(
            build_fortran_line(
                (int(psi_ind), int(meta["istepc"]), int(meta["npart"]), int(meta["max_class"])),
                int_width=8,
                reals=(
                    float(meta["b_min"]),
                    float(meta["b_max"]),
                    float(meta["bmref"]),
                    float(meta["coeps"]),
                    float(meta["y2"]),
                    float(meta["y3"]),
                ),
                real_width=20,
                real_digits=10,
                real_letter="E",
            )
            + "\n"
        )


def _write_diagnostic_bigint(
    *,
    bigint: Sequence[float],
    multra: int,
    hit_rat: int,
    nintfp: int,
    y2: float,
    y3: float,
    coeps: float,
    psi_ind: int,
    path_prefix: str = "",
) -> None:
    diag_path = f"{path_prefix}diagnostic_bigint.dat"
    with open(diag_path, "a", encoding="utf-8") as handle:
        handle.write(
            build_fortran_line(
                (int(psi_ind), int(multra), int(hit_rat), int(nintfp)),
                int_width=8,
                reals=(float(y2), float(y3), float(coeps), *[float(val) for val in bigint]),
                real_width=20,
                real_digits=10,
                real_letter="E",
            )
            + "\n"
        )


class DiagnosticLogger:
    def __init__(self, *, path_prefix: str = "") -> None:
        self.path_prefix = path_prefix
        self.diagnostic_path = f"{path_prefix}diagnostic.dat"
        self.trap_path = f"{path_prefix}diagnostic_first_trap.dat"
        self.istepc = 0
        self.max_class = 0
        self.first_trap_written = False
        self.snapshot_written = False
        with open(self.diagnostic_path, "w", encoding="utf-8") as handle:
            handle.write("")

    def callback(self, event_mask, icount, ipa, add_on) -> None:
        mask = np.asarray(event_mask)
        if not mask.any():
            return
        icount_np = np.asarray(icount)
        ipa_np = np.asarray(ipa)
        add_on_np = np.asarray(add_on)
        idxs = np.nonzero(mask)[0]
        with open(self.diagnostic_path, "a", encoding="utf-8") as handle:
            for idx in idxs:
                self.istepc += 1
                ipa_val = int(ipa_np[idx])
                self.max_class = max(self.max_class, ipa_val)
                handle.write(
                    build_fortran_line(
                        (int(idx + 1), int(icount_np[idx]), ipa_val),
                        int_width=8,
                        reals=(float(add_on_np[idx]),),
                        real_width=20,
                        real_digits=10,
                        real_letter="E",
                    )
                    + "\n"
                )

    def write_add(self, *, psi_ind: int, npart: int, meta: Dict[str, float]) -> None:
        add_path = f"{self.path_prefix}diagnostic_add.dat"
        with open(add_path, "w", encoding="utf-8") as handle:
            handle.write(
                build_fortran_line(
                    (int(psi_ind), int(self.istepc), int(npart), int(self.max_class)),
                    int_width=8,
                    reals=(
                        float(meta["b_min"]),
                        float(meta["b_max"]),
                        float(meta["bmref"]),
                        float(meta["coeps"]),
                        float(meta["y2"]),
                        float(meta["y3"]),
                    ),
                    real_width=20,
                    real_digits=10,
                    real_letter="E",
                )
                + "\n"
            )

    def write_bigint(
        self,
        *,
        psi_ind: int,
        multra: int,
        hit_rat: int,
        nintfp: int,
        y2: float,
        y3: float,
        coeps: float,
        bigint: Sequence[float],
    ) -> None:
        _write_diagnostic_bigint(
            bigint=bigint,
            multra=multra,
            hit_rat=hit_rat,
            nintfp=nintfp,
            y2=y2,
            y3=y3,
            coeps=coeps,
            psi_ind=psi_ind,
            path_prefix=self.path_prefix,
        )

    def trap_callback(self, event_mask, isw, iswst, p_i, p_h, icount, ipa, phi, j, step_index) -> None:
        if self.first_trap_written:
            return
        mask = np.asarray(event_mask)
        if not mask.any():
            return
        idxs = np.nonzero(mask)[0]
        first_idx = int(idxs[0])
        isw_np = np.asarray(isw)
        iswst_np = np.asarray(iswst)
        p_i_np = np.asarray(p_i)
        p_h_np = np.asarray(p_h)
        icount_np = np.asarray(icount)
        ipa_np = np.asarray(ipa)
        with open(self.trap_path, "w", encoding="utf-8") as handle:
            handle.write(f"# first_event_index={first_idx + 1}\n")
            handle.write(f"# phi={float(phi):.16e} n={int(step_index)} j1={int(j)}\n")
            handle.write("# columns: idx isw iswst icount ipa p_i p_h event_mask\n")
            for ii in range(p_i_np.shape[0]):
                handle.write(
                    f"{ii + 1:8d} {int(isw_np[ii]):8d} {int(iswst_np[ii]):8d}"
                    f" {int(icount_np[ii]):8d} {int(ipa_np[ii]):8d}"
                    f" {float(p_i_np[ii]):20.10e} {float(p_h_np[ii]):20.10e}"
                    f" {int(mask[ii]):8d}\n"
                )
        self.first_trap_written = True

    def snapshot_callback(self, isw, iswst, p_i, p_h, icount, ipa, phi, j, step_index) -> None:
        if self.snapshot_written:
            return
        isw_np = np.asarray(isw)
        iswst_np = np.asarray(iswst)
        p_i_np = np.asarray(p_i)
        p_h_np = np.asarray(p_h)
        icount_np = np.asarray(icount)
        ipa_np = np.asarray(ipa)
        event_mask = (isw_np == 2) & (iswst_np == 1)
        with open(self.trap_path.replace("diagnostic_first_trap", "diagnostic_snapshot"), "w", encoding="utf-8") as handle:
            handle.write(f"# phi={float(phi):.16e} n={int(step_index)} j1={int(j)}\n")
            handle.write("# columns: idx isw iswst icount ipa p_i p_h event_mask\n")
            for ii in range(p_i_np.shape[0]):
                handle.write(
                    f"{ii + 1:8d} {int(isw_np[ii]):8d} {int(iswst_np[ii]):8d}"
                    f" {int(icount_np[ii]):8d} {int(ipa_np[ii]):8d}"
                    f" {float(p_i_np[ii]):20.10e} {float(p_h_np[ii]):20.10e}"
                    f" {int(event_mask[ii]):8d}\n"
                )
        self.snapshot_written = True


def run_neo_from_boozer(
    booz: BoozerData,
    control: ControlParams,
    *,
    use_jax: bool = True,
    progress: bool = False,
    extension: str | None = None,
    legacy_mode: bool = False,
) -> NeoResults:
    if control.calc_cur != 0:
        raise NotImplementedError("calc_cur=1 is not implemented in NEO_JAX.")

    grid = prepare_grids(control.theta_n, control.phi_n, booz.nfp)
    legacy_writer = LegacyNeoWriter(extension=extension, progress=progress) if legacy_mode else None
    if legacy_writer is not None:
        legacy_writer.prepare_run()
        if control.write_output_files:
            legacy_writer.write_static_files(booz=booz, grid=grid)

    max_m_mode = control.max_m_mode if control.max_m_mode > 0 else int(np.max(np.abs(booz.ixm)))
    max_n_mode = control.max_n_mode if control.max_n_mode > 0 else int(np.max(np.abs(booz.ixn)))

    if control.fluxs_arr:
        if booz.rmnc.shape[0] == len(control.fluxs_arr):
            surf_indices = list(range(booz.rmnc.shape[0]))
        else:
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

    results: List[NeoSurfaceResult] = []
    r_eff = 0.0

    write_diagnostic = bool(control.write_diagnostic) or _env_flag("NEO_JAX_WRITE_DIAGNOSTIC")
    write_trap_debug = write_diagnostic or _env_flag("NEO_JAX_WRITE_TRAP_DEBUG")
    diag_backend = os.getenv("NEO_JAX_DIAGNOSTIC_BACKEND", "python").strip().lower()
    disable_jit = _env_flag("NEO_JAX_DISABLE_JIT")
    force_psi1 = _env_flag("NEO_JAX_DIAGNOSTIC_FORCE_PSI1")
    snapshot_n = os.getenv("NEO_JAX_SNAPSHOT_N")
    snapshot_j1 = os.getenv("NEO_JAX_SNAPSHOT_J1")
    diagnostic_snapshot = None
    if snapshot_n and snapshot_j1:
        try:
            diagnostic_snapshot = (int(snapshot_n), int(snapshot_j1))
        except ValueError:
            diagnostic_snapshot = None

    flint_bo_jax_fn = flint_bo_jax
    if use_jax and not write_diagnostic and not disable_jit:
        flint_bo_jax_fn = jax.jit(flint_bo_jax, static_argnames=("params",))

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
        if legacy_writer is not None and control.write_output_files:
            legacy_writer.write_surface_files(surface.fields)

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
            use_python_loop = bool(control.write_integrate)
            if write_diagnostic and diag_backend == "jax" and not use_python_loop:
                if progress:
                    print("NEO_JAX: write_diagnostic enabled; using JAX backend with diagnostic callback")
                logger = DiagnosticLogger()
                out = flint_bo_jax(
                    surface,
                    params,
                    env,
                    nfp=booz.nfp,
                    rt0=rt0,
                    diagnostic_callback=logger.callback,
                    diagnostic_trap_callback=logger.trap_callback if write_trap_debug else None,
                    diagnostic_snapshot=(
                        (diagnostic_snapshot[0] - 1, diagnostic_snapshot[1] - 1)
                        if diagnostic_snapshot
                        else None
                    ),
                    diagnostic_snapshot_callback=logger.snapshot_callback if diagnostic_snapshot else None,
                )
                jnp.asarray(out["y2"]).block_until_ready()
                jnp.asarray(out["bigint"]).block_until_ready()
                etamin = float(surface.b_min / surface.bmref)
                etamax = float(surface.b_max / surface.bmref)
                heta = (etamax - etamin) / (params.npart - 1)
                coeps = float(np.pi * rt0 * rt0 * heta / (8.0 * np.sqrt(2.0)))
                psi_ind_diag = int(local_idx + 1)
                if force_psi1 and (control.fluxs_arr is not None and len(control.fluxs_arr) == 1):
                    psi_ind_diag = 1
                logger.write_add(
                    psi_ind=psi_ind_diag,
                    npart=params.npart,
                    meta={
                        "b_min": float(surface.b_min),
                        "b_max": float(surface.b_max),
                        "bmref": float(surface.bmref),
                        "coeps": coeps,
                        "y2": float(out["y2"]),
                        "y3": float(out["y3"]),
                    },
                )
                logger.write_bigint(
                    psi_ind=psi_ind_diag,
                    multra=params.multra,
                    hit_rat=int(out["hit_rat"]),
                    nintfp=int(out["nintfp"]),
                    y2=float(out["y2"]),
                    y3=float(out["y3"]),
                    coeps=coeps,
                    bigint=np.asarray(out["bigint"]),
                )
            else:
                if write_diagnostic and progress:
                    print("NEO_JAX: write_diagnostic enabled; using Python-loop backend to emit diagnostic.dat")
                if write_diagnostic or use_python_loop:
                    out = flint_bo(
                        surface,
                        params,
                        env,
                        nfp=booz.nfp,
                        rt0=rt0,
                        diagnostic=write_diagnostic,
                        diagnostic_trap=write_trap_debug,
                        diagnostic_snapshot=diagnostic_snapshot,
                        collect_convergence=bool(control.write_integrate),
                    )
                else:
                    out = flint_bo_jax_fn(surface, params, env, nfp=booz.nfp, rt0=rt0)
        else:
            out = flint_bo(
                surface,
                params,
                env,
                nfp=booz.nfp,
                rt0=rt0,
                diagnostic=write_diagnostic,
                diagnostic_trap=write_trap_debug,
                diagnostic_snapshot=diagnostic_snapshot,
                collect_convergence=bool(control.write_integrate),
            )

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

        s = float(booz.es[surf_idx])
        if local_idx == 0:
            dpsi = s
        else:
            dpsi = s - float(booz.es[surf_indices[local_idx - 1]])
        r_eff = r_eff + float(out["drdpsi"] * dpsi)

        flux_index = control.fluxs_arr[local_idx] if control.fluxs_arr else surf_idx + 1
        result = NeoSurfaceResult(
            flux_index=flux_index,
            s=s,
            r_eff=r_eff,
            iota=float(booz.iota[surf_idx]),
            b_ref=b_ref,
            r_ref=r_ref,
            epsilon_effective=epstot,
            epsilon_effective_by_class=epspar,
            ctrone=float(out.get("ctrone", 0.0)),
            ctrtot=float(out.get("ctrtot", 0.0)),
            bareph=float(out.get("bareph", 0.0)),
            barept=float(out.get("barept", 0.0)),
            yps=float(out.get("yps", 0.0)),
            diagnostics=out,
        )

        if write_diagnostic:
            diagnostic_events = out.pop("diagnostic_events", None)
            diagnostic_meta = out.pop("diagnostic_meta", None)
            psi_ind_diag = int(local_idx + 1)
            if force_psi1 and (control.fluxs_arr is not None and len(control.fluxs_arr) == 1):
                psi_ind_diag = 1
            if diagnostic_events is not None and diagnostic_meta is not None:
                _write_diagnostic_files(
                    events=diagnostic_events,
                    meta=diagnostic_meta,
                    psi_ind=psi_ind_diag,
                )
                _write_diagnostic_bigint(
                    bigint=np.asarray(out["bigint"]),
                    multra=params.multra,
                    hit_rat=int(out["hit_rat"]),
                    nintfp=int(out["nintfp"]),
                    y2=float(out["y2"]),
                    y3=float(out["y3"]),
                    coeps=float(diagnostic_meta["coeps"]),
                    psi_ind=psi_ind_diag,
                )
                if progress:
                    print("NEO_JAX: wrote diagnostic.dat, diagnostic_add.dat, diagnostic_bigint.dat")

        if legacy_writer is not None and control.write_integrate:
            convergence_history = out.get("convergence_history")
            if convergence_history is not None:
                legacy_writer.write_conver(convergence_history)

        if legacy_writer is not None and extension is not None:
            legacy_writer.append_neolog(psi_ind=local_idx + 1, out=out, epstot=epstot)

        results.append(result)
        if progress:
            print(
                f"NEO_JAX: epstot={result['epstot']:.6e} reff={result['reff']:.6e} iota={result['iota']:.6e}"
            )

    return NeoResults(results)


def run_neo_from_boozmn(
    boozmn_path: str,
    control: ControlParams,
    *,
    use_jax: bool = True,
    progress: bool = False,
    extension: str | None = None,
    legacy_mode: bool = False,
) -> NeoResults:
    booz = read_boozmn(
        boozmn_path,
        max_m_mode=control.max_m_mode,
        max_n_mode=control.max_n_mode,
        fluxs_arr=control.fluxs_arr,
        extension=extension,
    )
    return run_neo_from_boozer(
        booz,
        control,
        use_jax=use_jax,
        progress=progress,
        extension=extension,
        legacy_mode=legacy_mode,
    )
