"""Integration routines for NEO_JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

import jax
import jax.numpy as jnp

from .geometry import neo_eval

Array = jax.Array

NPQ = 4


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RhsState:
    isw: Array
    ipa: Array
    icount: Array
    ipmax: Array
    pard0: Array

    def tree_flatten(self):
        return (self.isw, self.ipa, self.icount, self.ipmax, self.pard0), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class RhsEnv:
    splines: dict
    grid: dict
    eta: Array
    bmod0: Array
    iota: Array
    curr_pol: Array | None = None
    curr_tor: Array | None = None

    def tree_flatten(self):
        return (self.splines, self.grid, self.eta, self.bmod0, self.iota, self.curr_pol, self.curr_tor), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def rhs_bo1(phi: Array, y: Array, state: RhsState, env: RhsEnv) -> Tuple[Array, RhsState]:
    """Right-hand side for the field-line ODE (port of rhs_bo1.f90)."""
    theta = y[0]

    bmod, gval, geodcu, pardeb, _qval = neo_eval(
        theta,
        phi,
        env.splines["b_spl"],
        env.splines["g_spl"],
        env.splines["k_spl"],
        env.splines["p_spl"],
        env.splines.get("q_spl"),
        env.grid,
    )

    bmodm2 = 1.0 / (bmod * bmod)
    bmodm3 = bmodm2 / bmod
    bra = bmod / env.bmod0

    ipass = jnp.where((pardeb * state.pard0 <= 0) & (pardeb > 0), 1, 0).astype(state.isw.dtype)
    ipmax = jnp.where(
        (state.ipmax == 0) & (pardeb * state.pard0 <= 0) & (pardeb < 0),
        1,
        state.ipmax,
    ).astype(state.isw.dtype)

    pard0 = pardeb

    dery = jnp.zeros_like(y)
    dery = dery.at[0].set(env.iota)
    dery = dery.at[1].set(bmodm2)
    dery = dery.at[2].set(bmodm2 * gval)
    dery = dery.at[3].set(geodcu * bmodm3)

    eta_pos = env.eta > 0
    eta_safe = jnp.where(eta_pos, env.eta, jnp.asarray(1.0, dtype=env.eta.dtype))
    inv_eta = jnp.where(eta_pos, 1.0 / eta_safe, 0.0)
    sqeta = jnp.sqrt(eta_safe)
    inv_sqeta = jnp.where(eta_pos, 1.0 / sqeta, 0.0)

    subsq = 1.0 - bra * inv_eta
    mask = (subsq > 0) & eta_pos

    safe_subsq = jnp.where(mask, subsq, 0.0)
    sq = jnp.sqrt(safe_subsq) * bmodm2
    p_i = jnp.where(mask, sq, 0.0)
    p_h = jnp.where(mask, sq * (4.0 / bra - inv_eta) * geodcu * inv_sqeta, 0.0)

    # Update particle state
    one_i = jnp.array(1, dtype=state.isw.dtype)
    two_i = jnp.array(2, dtype=state.isw.dtype)
    zero_i = jnp.array(0, dtype=state.isw.dtype)
    isw = jnp.where(mask, one_i, jnp.where(state.isw == 1, two_i, jnp.where(state.isw == 2, two_i, zero_i)))
    icount = state.icount + mask.astype(state.icount.dtype)
    ipa = state.ipa + ipass * mask.astype(state.ipa.dtype)

    dery = dery.at[NPQ : NPQ + env.eta.shape[0]].set(p_i)
    dery = dery.at[NPQ + env.eta.shape[0] : NPQ + 2 * env.eta.shape[0]].set(p_h)

    new_state = RhsState(isw=isw, ipa=ipa, icount=icount, ipmax=ipmax, pard0=pard0)
    return dery, new_state


def rk4_step(phi: Array, y: Array, state: RhsState, env: RhsEnv, h: float) -> Tuple[Array, Array, RhsState]:
    """Run a single RK4 step, threading RHS state (port of rk4d_bo1.f90)."""
    hh = h / 2.0
    h6 = h / 6.0

    k1, state1 = rhs_bo1(phi, y, state, env)
    y1 = y + hh * k1

    k2, state2 = rhs_bo1(phi + hh, y1, state1, env)
    y2 = y + hh * k2

    k3, state3 = rhs_bo1(phi + hh, y2, state2, env)
    y3 = y + h * k3

    k4, state4 = rhs_bo1(phi + h, y3, state3, env)

    y_new = y + h6 * (k1 + k4 + 2.0 * (k2 + k3))
    phi_new = phi + h
    return phi_new, y_new, state4


def _process_trapped(
    state: RhsState,
    iswst: Array,
    p_i: Array,
    p_h: Array,
    bigint: Array,
    adimax: Array,
    multra: int,
) -> Tuple[RhsState, Array, Array, Array, Array, Array]:
    mask2 = state.isw == 2
    m_cl = jnp.clip(state.ipa, 1, multra).astype(state.ipa.dtype)

    def body(i, carry):
        bigint_acc, adimax_acc = carry

        def add_fn(carry):
            bigint_acc, adimax_acc = carry
            safe_pi = jnp.where(p_i[i] == 0, jnp.array(1.0, dtype=p_i.dtype), p_i[i])
            add_on = (p_h[i] * p_h[i]) / safe_pi * iswst[i]
            idx = m_cl[i] - 1
            bigint_acc = bigint_acc.at[idx].add(add_on)
            adimax_acc = jnp.where(state.ipa[i] == 1, p_i[i], adimax_acc)
            return bigint_acc, adimax_acc

        return jax.lax.cond(mask2[i], add_fn, lambda c: c, carry)

    bigint, adimax = jax.lax.fori_loop(0, p_i.shape[0], body, (bigint, adimax))

    iswst = jnp.where(mask2, 1, iswst)
    p_h = jnp.where(mask2, 0.0, p_h)
    p_i = jnp.where(mask2, 0.0, p_i)
    zero_int = jnp.zeros_like(state.isw)
    isw = jnp.where(mask2, zero_int, state.isw)
    icount = jnp.where(mask2, zero_int, state.icount)
    ipa = jnp.where(mask2, zero_int, state.ipa)
    state = RhsState(isw, ipa, icount, state.ipmax, state.pard0)

    return state, iswst, p_i, p_h, bigint, adimax


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class FlintParams:
    npart: int
    multra: int
    nstep_per: int
    nstep_min: int
    nstep_max: int
    acc_req: float
    no_bins: int
    calc_nstep_max: int

    def tree_flatten(self):
        return (
            self.npart,
            self.multra,
            self.nstep_per,
            self.nstep_min,
            self.nstep_max,
            self.acc_req,
            self.no_bins,
            self.calc_nstep_max,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def flint_bo(
    surface,
    params: FlintParams,
    env: RhsEnv,
    nfp: int,
    rt0: float | None = None,
    *,
    Rmajor: float | None = None,
    diagnostic: bool = False,
    diagnostic_trap: bool = False,
    diagnostic_trap_path: str = "diagnostic_first_trap.dat",
    diagnostic_snapshot: tuple[int, int] | None = None,
    diagnostic_snapshot_path: str = "diagnostic_snapshot.dat",
    collect_convergence: bool = False,
):
    """Python-loop port of flint_bo.f90 (not yet JIT-optimized)."""
    if rt0 is None:
        if Rmajor is None:
            raise ValueError("Either rt0 or Rmajor must be provided.")
        rt0 = float(Rmajor)
    elif Rmajor is not None and float(Rmajor) != float(rt0):
        raise ValueError("rt0 and Rmajor must match if both are provided.")
    npart = params.npart
    multra = params.multra
    ndim = NPQ + 2 * npart

    # Particle grids
    etamin = surface.b_min / surface.bmref
    etamax = surface.b_max / surface.bmref
    heta = (etamax - etamin) / (npart - 1)
    etamin = etamin + heta / 2.0
    eta = etamin + heta * jnp.arange(npart)

    # Override env with surface-specific eta and bmod0
    env = RhsEnv(
        splines=env.splines,
        grid=env.grid,
        eta=eta,
        bmod0=surface.bmref,
        iota=env.iota,
        curr_pol=env.curr_pol,
        curr_tor=env.curr_tor,
    )

    coeps = jnp.pi * rt0 * rt0 * heta / (8.0 * jnp.sqrt(2.0))
    if env.curr_pol is None or env.curr_tor is None:
        j_iota_i = 0.0
    else:
        j_iota_i = env.curr_pol + env.iota * env.curr_tor

    # Initial state
    y = jnp.zeros(ndim)
    y = y.at[0].set(surface.theta_bmax)
    phi = surface.phi_bmax

    state = RhsState(
        isw=jnp.zeros(npart, dtype=jnp.int32),
        ipa=jnp.zeros(npart, dtype=jnp.int32),
        icount=jnp.zeros(npart, dtype=jnp.int32),
        ipmax=jnp.array(0, dtype=jnp.int32),
        pard0=jnp.array(0.0),
    )

    # Initialize pard0
    _bmod, _gval, _geodcu, pard0, _qval = neo_eval(
        surface.theta_bmax,
        surface.phi_bmax,
        env.splines["b_spl"],
        env.splines["g_spl"],
        env.splines["k_spl"],
        env.splines["p_spl"],
        env.splines.get("q_spl"),
        env.grid,
    )
    state = RhsState(state.isw, state.ipa, state.icount, state.ipmax, pard0)

    # Main accumulators
    bigint = jnp.zeros(multra)
    adimax = jnp.array(0.0)
    aditot = jnp.array(0.0)

    iswst = jnp.zeros(npart, dtype=jnp.int32)

    # Integration parameters
    nper = nfp
    hphi = 2.0 * jnp.pi / (params.nstep_per * nper)

    nstep_max_c = params.nstep_max
    exist_first_ratfl = 0
    hit_rat = 0
    nfp_rat = 0
    nfl_rat = 0
    delta_theta_rat = 0.0

    theta0 = surface.theta_bmax
    phi0 = surface.phi_bmax

    theta_d_min = 2.0 * jnp.pi
    n_iota = 1
    m_iota = 1
    iota_bar_fp = 0.0
    n_gap = 0

    diagnostic_events = [] if diagnostic else None
    max_class = 0
    trap_written = False
    snapshot_written = False
    convergence_history = [] if collect_convergence else None

    # Main loop
    for n in range(1, params.nstep_max + 1):
        for _j1 in range(1, params.nstep_per + 1):
            phi, y, state = rk4_step(phi, y, state, env, float(hphi))

            # Process trapped particle contributions
            p_i = y[NPQ : NPQ + npart]
            p_h = y[NPQ + npart : NPQ + 2 * npart]

            if diagnostic:
                mask2 = np.asarray(state.isw == 2)
                if mask2.any():
                    iswst_np = np.asarray(iswst)
                    event_mask = mask2 & (iswst_np == 1)
                    if event_mask.any():
                        icount_np = np.asarray(state.icount)
                        ipa_np = np.asarray(state.ipa)
                        p_i_np = np.asarray(p_i)
                        p_h_np = np.asarray(p_h)
                        safe_pi = np.where(p_i_np == 0, 1.0, p_i_np)
                        add_on_np = (p_h_np * p_h_np) / safe_pi * iswst_np
                        idxs = np.nonzero(event_mask)[0]
                        for idx in idxs:
                            diagnostic_events.append(
                                (int(idx + 1), int(icount_np[idx]), int(ipa_np[idx]), float(add_on_np[idx]))
                            )
                        max_class = max(max_class, int(np.max(ipa_np[event_mask])))
                        if diagnostic_trap and not trap_written:
                            first_idx = int(idxs[0])
                            with open(diagnostic_trap_path, "w", encoding="utf-8") as handle:
                                handle.write(f"# first_event_index={first_idx + 1}\n")
                                handle.write(f"# phi={float(phi):.16e} n={n} j1={_j1}\n")
                                handle.write(
                                    "# columns: idx isw iswst icount ipa p_i p_h event_mask\n"
                                )
                                isw_np = np.asarray(state.isw)
                                for ii in range(p_i_np.shape[0]):
                                    handle.write(
                                        f"{ii + 1:8d} {int(isw_np[ii]):8d} {int(iswst_np[ii]):8d}"
                                        f" {int(icount_np[ii]):8d} {int(ipa_np[ii]):8d}"
                                        f" {float(p_i_np[ii]):20.10e} {float(p_h_np[ii]):20.10e}"
                                        f" {int(event_mask[ii]):8d}\n"
                                    )
                            trap_written = True
            if diagnostic_snapshot and not snapshot_written:
                snap_n, snap_j1 = diagnostic_snapshot
                if n == snap_n and _j1 == snap_j1:
                    mask2 = np.asarray(state.isw == 2)
                    iswst_np = np.asarray(iswst)
                    event_mask = mask2 & (iswst_np == 1)
                    icount_np = np.asarray(state.icount)
                    ipa_np = np.asarray(state.ipa)
                    p_i_np = np.asarray(p_i)
                    p_h_np = np.asarray(p_h)
                    isw_np = np.asarray(state.isw)
                    with open(diagnostic_snapshot_path, "w", encoding="utf-8") as handle:
                        handle.write(f"# phi={float(phi):.16e} n={n} j1={_j1}\n")
                        handle.write("# columns: idx isw iswst icount ipa p_i p_h event_mask\n")
                        for ii in range(p_i_np.shape[0]):
                            handle.write(
                                f"{ii + 1:8d} {int(isw_np[ii]):8d} {int(iswst_np[ii]):8d}"
                                f" {int(icount_np[ii]):8d} {int(ipa_np[ii]):8d}"
                                f" {float(p_i_np[ii]):20.10e} {float(p_h_np[ii]):20.10e}"
                                f" {int(event_mask[ii]):8d}\n"
                            )
                    snapshot_written = True

            state, iswst, p_i, p_h, bigint, adimax = _process_trapped(
                state, iswst, p_i, p_h, bigint, adimax, multra
            )

            y = y.at[NPQ : NPQ + npart].set(p_i)
            y = y.at[NPQ + npart : NPQ + 2 * npart].set(p_h)

            if int(state.ipmax) == 1:
                aditot = aditot + adimax
                state = RhsState(state.isw, state.ipa, state.icount, jnp.array(0, dtype=jnp.int32), state.pard0)

        if collect_convergence and convergence_history is not None:
            epstot_check = 0.0
            for m_cl in range(1, multra + 1):
                epspar_check = float(coeps * bigint[m_cl - 1] * y[1] / (y[2] * y[2]))
                epstot_check = epstot_check + epspar_check
            convergence_history.append(
                (
                    float(n),
                    epstot_check,
                    float(y[3]),
                    float(y[NPQ + npart - 1] / y[1]),
                    float(aditot / y[1]),
                )
            )

        # Rational surface detection
        theta = y[0]
        if n <= params.nstep_min:
            theta_rs = theta - theta0
            if n == 1:
                theta_iota = theta_rs
                iota_bar_fp = float(theta_iota / (2.0 * jnp.pi))

            m = int(jnp.floor(theta_rs / (2.0 * jnp.pi)))
            theta_rs = theta_rs - m * 2.0 * jnp.pi
            theta_d = theta_rs if theta_rs <= jnp.pi else theta_rs - 2.0 * jnp.pi

            if abs(theta_d) < abs(theta_d_min):
                theta_d_min = theta_d
                n_iota = n
                if theta_d >= 0:
                    m_iota = m
                else:
                    m_iota = m + 1

        if n == params.nstep_min:
            theta_gap = 2.0 * jnp.pi / n_iota
            n_gap = int(n_iota * int(abs(theta_gap / theta_d_min)))
            if n_gap > params.nstep_min:
                nstep_max_c = n_gap
            else:
                nstep_max_c = int(n_gap * jnp.ceil(params.nstep_min / n_gap))

            if nstep_max_c > params.nstep_max:
                hit_rat = 1
                nfp_rat = int(jnp.ceil(1.0 / params.acc_req / iota_bar_fp)) if iota_bar_fp != 0 else 0
                if nfp_rat % n_iota != 0:
                    nfp_rat = nfp_rat + n_iota - (nfp_rat % n_iota)
                if nfp_rat >= params.nstep_min:
                    exist_first_ratfl = 1
                    nstep_max_c = nfp_rat
                nfl_rat = int(jnp.ceil(params.no_bins / n_iota))
                delta_theta_rat = float(theta_gap / (nfl_rat + 1))
                if params.calc_nstep_max == 1:
                    hit_rat = 0
                if hit_rat == 1 and exist_first_ratfl == 0:
                    break

        if params.calc_nstep_max == 0 and n == nstep_max_c:
            break

    nintfp = n
    y2 = y[1]
    y3 = y[2]
    y4 = y[3]
    y3npart = y[NPQ + npart - 1]

    # Rational surface correction
    if hit_rat == 1:
        if exist_first_ratfl == 0:
            if collect_convergence and convergence_history is not None:
                convergence_history = []
            bigint = jnp.zeros(multra)
            adimax = jnp.array(0.0)
            aditot = jnp.array(0.0)
            y2 = jnp.array(0.0)
            y3 = jnp.array(0.0)
            y4 = jnp.array(0.0)
            y3npart = jnp.array(0.0)

        for nfl in range(exist_first_ratfl, nfl_rat + 1):
            bigint_s = jnp.zeros(multra)
            adimax_s = jnp.array(0.0)
            aditot_s = jnp.array(0.0)

            iswst = jnp.zeros(npart, dtype=jnp.int32)
            state = RhsState(
                isw=jnp.zeros(npart, dtype=jnp.int32),
                ipa=jnp.zeros(npart, dtype=jnp.int32),
                icount=jnp.zeros(npart, dtype=jnp.int32),
                ipmax=jnp.array(0, dtype=jnp.int32),
                pard0=state.pard0,
            )

            phi = phi0
            y = jnp.zeros(ndim)
            theta = theta0 + nfl * delta_theta_rat
            y = y.at[0].set(theta)

            for _n in range(1, nfp_rat + 1):
                # Update pard0 at the start of each field period
                _bmod, _gval, _geodcu, pard0, _qval = neo_eval(
                    y[0],
                    phi,
                    env.splines["b_spl"],
                    env.splines["g_spl"],
                    env.splines["k_spl"],
                    env.splines["p_spl"],
                    env.splines.get("q_spl"),
                    env.grid,
                )
                state = RhsState(state.isw, state.ipa, state.icount, state.ipmax, pard0)

                for _j1 in range(1, params.nstep_per + 1):
                    phi, y, state = rk4_step(phi, y, state, env, float(hphi))

                    p_i = y[NPQ : NPQ + npart]
                    p_h = y[NPQ + npart : NPQ + 2 * npart]

                    state, iswst, p_i, p_h, bigint_s, adimax_s = _process_trapped(
                        state, iswst, p_i, p_h, bigint_s, adimax_s, multra
                    )

                    y = y.at[NPQ : NPQ + npart].set(p_i)
                    y = y.at[NPQ + npart : NPQ + 2 * npart].set(p_h)

                    if int(state.ipmax) == 1:
                        aditot_s = aditot_s + adimax_s
                        state = RhsState(state.isw, state.ipa, state.icount, jnp.array(0, dtype=jnp.int32), state.pard0)

                if collect_convergence and convergence_history is not None:
                    epstot_check = 0.0
                    for m_cl in range(1, multra + 1):
                        epspar_check = float(coeps * bigint_s[m_cl - 1] * y[1] / (y[2] * y[2]))
                        epstot_check = epstot_check + epspar_check
                    convergence_history.append(
                        (
                            float(nfl * nfp_rat + _n),
                            epstot_check,
                            float(y[3]),
                            float(y[NPQ + npart - 1] / y[1]),
                            float(aditot_s / y[1]),
                        )
                    )

            y2_s = y[1]
            y3_s = y[2]
            y4_s = y[3]
            y3npart_s = y[NPQ + npart - 1]

            bigint = bigint + bigint_s
            aditot = aditot + aditot_s
            y2 = y2 + y2_s
            y3 = y3 + y3_s
            y4 = y4 + y4_s
            y3npart = y3npart + y3npart_s

        n = nfp_rat * (nfl_rat + 1)

    # Final results
    epspar = jnp.zeros(multra)
    epstot = jnp.array(0.0)
    for m_cl in range(1, multra + 1):
        epspar = epspar.at[m_cl - 1].set(coeps * bigint[m_cl - 1] * y2 / (y3 * y3))
        epstot = epstot + epspar[m_cl - 1]

    ctrone = aditot / y2
    ctrtot = y3npart / y2

    bareph = (jnp.pi * ctrone) ** 2 / 8.0
    barept = (jnp.pi * ctrtot) ** 2 / 8.0

    drdpsi = y2 / y3
    yps = y4 * j_iota_i

    out = {
        "epspar": epspar,
        "epstot": epstot,
        "ctrone": ctrone,
        "ctrtot": ctrtot,
        "bareph": bareph,
        "barept": barept,
        "drdpsi": drdpsi,
        "yps": yps,
        "y2": y2,
        "y3": y3,
        "y4": y4,
        "y3npart": y3npart,
        "bigint": bigint,
        "nintfp": nintfp,
        "hit_rat": hit_rat,
        "n_iota": n_iota,
        "m_iota": m_iota,
        "n_gap": n_gap,
        "final_n": n,
        "nfp_rat": nfp_rat,
        "nfl_rat": nfl_rat,
    }
    if collect_convergence and convergence_history is not None:
        out["convergence_history"] = convergence_history
    if diagnostic and diagnostic_events is not None:
        out["diagnostic_events"] = diagnostic_events
        out["diagnostic_meta"] = {
            "istepc": len(diagnostic_events),
            "max_class": max_class,
            "b_min": float(surface.b_min),
            "b_max": float(surface.b_max),
            "bmref": float(surface.bmref),
            "coeps": float(coeps),
            "y2": float(y2),
            "y3": float(y3),
            "npart": int(npart),
        }

    return out


def flint_bo_jax(
    surface,
    params: FlintParams,
    env: RhsEnv,
    nfp: int,
    rt0: float | None = None,
    *,
    Rmajor: float | None = None,
    diagnostic_callback=None,
    diagnostic_trap_callback=None,
    diagnostic_snapshot: tuple[int, int] | None = None,
    diagnostic_snapshot_callback=None,
    convergence_callback=None,
    convergence_reset_callback=None,
    strict_parity: bool = False,
):
    """JAX-friendly integration loop with rational-surface correction."""
    if rt0 is None:
        if Rmajor is None:
            raise ValueError("Either rt0 or Rmajor must be provided.")
        rt0 = float(Rmajor)
    elif Rmajor is not None and float(Rmajor) != float(rt0):
        raise ValueError("rt0 and Rmajor must match if both are provided.")
    npart = params.npart
    multra = params.multra
    ndim = NPQ + 2 * npart

    # Particle grids
    etamin = surface.b_min / surface.bmref
    etamax = surface.b_max / surface.bmref
    heta = (etamax - etamin) / (npart - 1)
    etamin = etamin + heta / 2.0
    eta = etamin + heta * jnp.arange(npart)

    env = RhsEnv(
        splines=env.splines,
        grid=env.grid,
        eta=eta,
        bmod0=surface.bmref,
        iota=env.iota,
        curr_pol=env.curr_pol,
        curr_tor=env.curr_tor,
    )

    coeps = jnp.pi * rt0 * rt0 * heta / (8.0 * jnp.sqrt(2.0))
    if env.curr_pol is None or env.curr_tor is None:
        j_iota_i = 0.0
    else:
        j_iota_i = env.curr_pol + env.iota * env.curr_tor

    y = jnp.zeros(ndim)
    y = y.at[0].set(surface.theta_bmax)
    phi0 = surface.phi_bmax
    phi = phi0

    state = RhsState(
        isw=jnp.zeros(npart, dtype=jnp.int32),
        ipa=jnp.zeros(npart, dtype=jnp.int32),
        icount=jnp.zeros(npart, dtype=jnp.int32),
        ipmax=jnp.array(0, dtype=jnp.int32),
        pard0=jnp.array(0.0),
    )

    _bmod, _gval, _geodcu, pard0, _qval = neo_eval(
        surface.theta_bmax,
        surface.phi_bmax,
        env.splines["b_spl"],
        env.splines["g_spl"],
        env.splines["k_spl"],
        env.splines["p_spl"],
        env.splines.get("q_spl"),
        env.grid,
    )
    state = RhsState(state.isw, state.ipa, state.icount, state.ipmax, pard0)

    iswst = jnp.zeros(npart, dtype=jnp.int32)
    bigint = jnp.zeros(multra)
    adimax = jnp.array(0.0)
    aditot = jnp.array(0.0)

    hphi = 2.0 * jnp.pi / (params.nstep_per * nfp)
    theta0 = surface.theta_bmax

    theta_d_min = 2.0 * jnp.pi
    n_iota = jnp.array(1, dtype=jnp.int32)
    m_iota = jnp.array(1, dtype=jnp.int32)
    iota_bar_fp = jnp.array(0.0)

    nstep_max_c = jnp.array(params.nstep_max, dtype=jnp.int32)
    hit_rat = jnp.array(0, dtype=jnp.int32)
    exist_first_ratfl = jnp.array(0, dtype=jnp.int32)
    nfp_rat = jnp.array(0, dtype=jnp.int32)
    nfl_rat = jnp.array(0, dtype=jnp.int32)
    delta_theta_rat = jnp.array(0.0)
    n_gap = jnp.array(0, dtype=jnp.int32)

    stop = jnp.array(False)
    n = jnp.array(0, dtype=jnp.int32)

    if diagnostic_snapshot is None:
        snap_n = None
        snap_j = None
    else:
        snap_n = jnp.array(diagnostic_snapshot[0], dtype=jnp.int32)
        snap_j = jnp.array(diagnostic_snapshot[1], dtype=jnp.int32)

    def integrate_period(carry, emit_diag: bool, step_index):
        phi, y, state, iswst, bigint, adimax, aditot = carry

        def rhs_inline(phi_local: Array, y_local: Array, state_local: RhsState) -> Tuple[Array, RhsState]:
            """Inline RHS evaluation to help XLA fuse neo_eval + RK4."""
            theta = y_local[0]

            bmod, gval, geodcu, pardeb, _qval = neo_eval(
                theta,
                phi_local,
                env.splines["b_spl"],
                env.splines["g_spl"],
                env.splines["k_spl"],
                env.splines["p_spl"],
                env.splines.get("q_spl"),
                env.grid,
            )

            bmodm2 = 1.0 / (bmod * bmod)
            bmodm3 = bmodm2 / bmod
            bra = bmod / env.bmod0

            ipass = jnp.where(
                (pardeb * state_local.pard0 <= 0) & (pardeb > 0), 1, 0
            ).astype(state_local.isw.dtype)
            ipmax = jnp.where(
                (state_local.ipmax == 0) & (pardeb * state_local.pard0 <= 0) & (pardeb < 0),
                1,
                state_local.ipmax,
            ).astype(state_local.isw.dtype)

            pard0 = pardeb

            dery = jnp.zeros_like(y_local)
            dery = dery.at[0].set(env.iota)
            dery = dery.at[1].set(bmodm2)
            dery = dery.at[2].set(bmodm2 * gval)
            dery = dery.at[3].set(geodcu * bmodm3)

            eta_pos = env.eta > 0
            eta_safe = jnp.where(eta_pos, env.eta, jnp.asarray(1.0, dtype=env.eta.dtype))
            inv_eta = jnp.where(eta_pos, 1.0 / eta_safe, 0.0)
            sqeta = jnp.sqrt(eta_safe)
            inv_sqeta = jnp.where(eta_pos, 1.0 / sqeta, 0.0)

            subsq = 1.0 - bra * inv_eta
            mask = (subsq > 0) & eta_pos

            safe_subsq = jnp.where(mask, subsq, 0.0)
            sq = jnp.sqrt(safe_subsq) * bmodm2
            p_i = jnp.where(mask, sq, 0.0)
            p_h = jnp.where(mask, sq * (4.0 / bra - inv_eta) * geodcu * inv_sqeta, 0.0)

            one_i = jnp.array(1, dtype=state_local.isw.dtype)
            two_i = jnp.array(2, dtype=state_local.isw.dtype)
            zero_i = jnp.array(0, dtype=state_local.isw.dtype)
            isw = jnp.where(
                mask,
                one_i,
                jnp.where(
                    state_local.isw == 1,
                    two_i,
                    jnp.where(state_local.isw == 2, two_i, zero_i),
                ),
            )
            icount = state_local.icount + mask.astype(state_local.icount.dtype)
            ipa = state_local.ipa + ipass * mask.astype(state_local.ipa.dtype)

            dery = dery.at[NPQ : NPQ + env.eta.shape[0]].set(p_i)
            dery = dery.at[NPQ + env.eta.shape[0] : NPQ + 2 * env.eta.shape[0]].set(p_h)

            new_state = RhsState(isw=isw, ipa=ipa, icount=icount, ipmax=ipmax, pard0=pard0)
            return dery, new_state

        def rk4_step_inline(
            phi_local: Array, y_local: Array, state_local: RhsState
        ) -> Tuple[Array, Array, RhsState]:
            """Inline RK4 to increase fusion with neo_eval."""
            hh = hphi / 2.0
            h6 = hphi / 6.0

            k1, state1 = rhs_inline(phi_local, y_local, state_local)
            y1 = y_local + hh * k1

            k2, state2 = rhs_inline(phi_local + hh, y1, state1)
            y2 = y_local + hh * k2

            k3, state3 = rhs_inline(phi_local + hh, y2, state2)
            y3 = y_local + hphi * k3

            k4, state4 = rhs_inline(phi_local + hphi, y3, state3)

            y_new = y_local + h6 * (k1 + k4 + 2.0 * (k2 + k3))
            phi_new = phi_local + hphi
            return phi_new, y_new, state4

        def inner_step(j, inner):
            phi, y, state, iswst, bigint, adimax, aditot = inner
            if strict_parity:
                phi, y, state = rk4_step(phi, y, state, env, hphi)
            else:
                phi, y, state = rk4_step_inline(phi, y, state)
            p_i = y[NPQ : NPQ + npart]
            p_h = y[NPQ + npart : NPQ + 2 * npart]
            if diagnostic_callback is not None and emit_diag:
                mask2 = state.isw == 2
                event_mask = mask2 & (iswst == 1)
                safe_pi = jnp.where(p_i == 0, jnp.array(1.0, dtype=p_i.dtype), p_i)
                add_on = (p_h * p_h) / safe_pi * iswst

                def _emit(_):
                    jax.debug.callback(diagnostic_callback, event_mask, state.icount, state.ipa, add_on, ordered=True)
                    return None

                _ = jax.lax.cond(jnp.any(event_mask), _emit, lambda _: None, operand=None)
            if diagnostic_trap_callback is not None and emit_diag:
                mask2 = state.isw == 2
                event_mask = mask2 & (iswst == 1)

                def _emit_trap(_):
                    jax.debug.callback(
                        diagnostic_trap_callback,
                        event_mask,
                        state.isw,
                        iswst,
                        p_i,
                        p_h,
                        state.icount,
                        state.ipa,
                        phi,
                        j,
                        step_index,
                        ordered=True,
                    )
                    return None

                _ = jax.lax.cond(jnp.any(event_mask), _emit_trap, lambda _: None, operand=None)
            if diagnostic_snapshot_callback is not None and emit_diag and snap_n is not None and snap_j is not None:
                def _emit_snap(_):
                    jax.debug.callback(
                        diagnostic_snapshot_callback,
                        state.isw,
                        iswst,
                        p_i,
                        p_h,
                        state.icount,
                        state.ipa,
                        phi,
                        j,
                        step_index,
                        ordered=True,
                    )
                    return None

                snap_cond = (step_index == snap_n) & (j == snap_j)
                _ = jax.lax.cond(snap_cond, _emit_snap, lambda _: None, operand=None)

            if strict_parity:
                state, iswst, p_i, p_h, bigint, adimax = _process_trapped(
                    state, iswst, p_i, p_h, bigint, adimax, multra
                )
            else:
                # Inline trapped-particle update to improve fusion in scan body.
                mask2 = state.isw == 2
                m_cl = jnp.clip(state.ipa, 1, multra).astype(state.ipa.dtype)

                def body(i, carry):
                    bigint_acc, adimax_acc = carry

                    def add_fn(carry):
                        bigint_acc, adimax_acc = carry
                        safe_pi = jnp.where(p_i[i] == 0, jnp.array(1.0, dtype=p_i.dtype), p_i[i])
                        add_on = (p_h[i] * p_h[i]) / safe_pi * iswst[i]
                        idx = m_cl[i] - 1
                        bigint_acc = bigint_acc.at[idx].add(add_on)
                        adimax_acc = jnp.where(state.ipa[i] == 1, p_i[i], adimax_acc)
                        return bigint_acc, adimax_acc

                    return jax.lax.cond(mask2[i], add_fn, lambda c: c, carry)

                bigint, adimax = jax.lax.fori_loop(0, p_i.shape[0], body, (bigint, adimax))

                iswst = jnp.where(mask2, 1, iswst)
                p_h = jnp.where(mask2, 0.0, p_h)
                p_i = jnp.where(mask2, 0.0, p_i)
                zero_int = jnp.zeros_like(state.isw)
                isw = jnp.where(mask2, zero_int, state.isw)
                icount = jnp.where(mask2, zero_int, state.icount)
                ipa = jnp.where(mask2, zero_int, state.ipa)
                state = RhsState(isw, ipa, icount, state.ipmax, state.pard0)

            y = y.at[NPQ : NPQ + npart].set(p_i)
            y = y.at[NPQ + npart : NPQ + 2 * npart].set(p_h)

            aditot = jnp.where(state.ipmax == 1, aditot + adimax, aditot)
            ipmax = jnp.where(
                state.ipmax == 1, jnp.array(0, dtype=state.ipmax.dtype), state.ipmax
            )
            state = RhsState(state.isw, state.ipa, state.icount, ipmax, state.pard0)
            return (phi, y, state, iswst, bigint, adimax, aditot)

        return jax.lax.fori_loop(0, params.nstep_per, inner_step, carry)

    def convergence_row(n_val, bigint_val, y_val, aditot_val):
        epspar_check = coeps * bigint_val * y_val[1] / (y_val[2] * y_val[2])
        epstot_check = jnp.sum(epspar_check)
        return jnp.asarray(
            [
                n_val.astype(y_val.dtype),
                epstot_check,
                y_val[3],
                y_val[NPQ + npart - 1] / y_val[1],
                aditot_val / y_val[1],
            ]
        )

    def emit_convergence(n_val, bigint_val, y_val, aditot_val):
        if convergence_callback is None:
            return
        row = convergence_row(n_val, bigint_val, y_val, aditot_val)
        jax.debug.callback(convergence_callback, row, ordered=True)

    def update_theta_min(n_val, theta, theta_d_min, n_iota, m_iota, iota_bar_fp):
        twopi = 2.0 * jnp.pi

        def body(_):
            theta_rs = theta - theta0
            iota_bar_fp_new = jnp.where(n_val == 1, theta_rs / twopi, iota_bar_fp)
            m = jnp.floor(theta_rs / twopi)
            theta_rs_mod = theta_rs - m * twopi
            theta_d = jnp.where(theta_rs_mod <= jnp.pi, theta_rs_mod, theta_rs_mod - twopi)
            update = jnp.abs(theta_d) < jnp.abs(theta_d_min)
            theta_d_min_new = jnp.where(update, theta_d, theta_d_min)
            n_iota_new = jnp.where(update, n_val, n_iota)
            m_iota_new = jnp.where(
                update, jnp.where(theta_d >= 0, m.astype(jnp.int32), (m + 1).astype(jnp.int32)), m_iota
            )
            return theta_d_min_new, n_iota_new, m_iota_new, iota_bar_fp_new

        return jax.lax.cond(
            n_val <= params.nstep_min, body, lambda _: (theta_d_min, n_iota, m_iota, iota_bar_fp), operand=None
        )

    def update_rational(
        n_val,
        theta_d_min,
        n_iota,
        iota_bar_fp,
        nstep_max_c,
        hit_rat,
        exist_first_ratfl,
        nfp_rat,
        nfl_rat,
        delta_theta_rat,
        n_gap,
    ):
        twopi = 2.0 * jnp.pi

        def body(_):
            theta_d_min_safe = jnp.where(theta_d_min == 0.0, 1.0e-12, theta_d_min)
            theta_gap = twopi / n_iota
            n_gap_new = n_iota * jnp.floor(jnp.abs(theta_gap / theta_d_min_safe)).astype(jnp.int32)
            nstep_max_c_new = jnp.where(
                n_gap_new > params.nstep_min,
                n_gap_new,
                n_gap_new * jnp.ceil(params.nstep_min / n_gap_new).astype(jnp.int32),
            )

            hit_rat_new = jnp.where(nstep_max_c_new > params.nstep_max, 1, 0).astype(jnp.int32)
            nfp_rat_new = jnp.where(
                (hit_rat_new == 1) & (iota_bar_fp != 0.0),
                jnp.ceil(1.0 / params.acc_req / iota_bar_fp).astype(jnp.int32),
                0,
            )
            nfp_rat_new = jnp.where(
                (hit_rat_new == 1) & (nfp_rat_new % n_iota != 0),
                nfp_rat_new + n_iota - (nfp_rat_new % n_iota),
                nfp_rat_new,
            )

            exist_first_ratfl_new = jnp.where(nfp_rat_new >= params.nstep_min, 1, 0).astype(jnp.int32)
            nstep_max_c_new = jnp.where(exist_first_ratfl_new == 1, nfp_rat_new, nstep_max_c_new)

            nfl_rat_new = jnp.where(
                hit_rat_new == 1,
                jnp.ceil(params.no_bins / n_iota).astype(jnp.int32),
                nfl_rat,
            )
            delta_theta_rat_new = jnp.where(
                hit_rat_new == 1,
                theta_gap / (nfl_rat_new + 1),
                delta_theta_rat,
            )

            hit_rat_new = jnp.where(params.calc_nstep_max == 1, 0, hit_rat_new).astype(jnp.int32)

            return (
                nstep_max_c_new,
                hit_rat_new,
                exist_first_ratfl_new,
                nfp_rat_new,
                nfl_rat_new,
                delta_theta_rat_new,
                n_gap_new,
            )

        return jax.lax.cond(
            n_val == params.nstep_min,
            body,
            lambda _: (nstep_max_c, hit_rat, exist_first_ratfl, nfp_rat, nfl_rat, delta_theta_rat, n_gap),
            operand=None,
        )

    def scan_body(carry, _):
        (
            phi,
            y,
            state,
            iswst,
            bigint,
            adimax,
            aditot,
            n,
            theta_d_min,
            n_iota,
            m_iota,
            iota_bar_fp,
            nstep_max_c,
            hit_rat,
            exist_first_ratfl,
            nfp_rat,
            nfl_rat,
            delta_theta_rat,
            n_gap,
            stop,
        ) = carry

        def do_step(_carry):
            (
                phi,
                y,
                state,
                iswst,
                bigint,
                adimax,
                aditot,
                n,
                theta_d_min,
                n_iota,
                m_iota,
                iota_bar_fp,
                nstep_max_c,
                hit_rat,
                exist_first_ratfl,
                nfp_rat,
                nfl_rat,
                delta_theta_rat,
                n_gap,
                stop,
            ) = _carry

            phi, y, state, iswst, bigint, adimax, aditot = integrate_period(
                (phi, y, state, iswst, bigint, adimax, aditot),
                True,
                n,
            )
            n_new = n + 1
            emit_convergence(n_new.astype(y.dtype), bigint, y, aditot)
            theta = y[0]
            theta_d_min_new, n_iota_new, m_iota_new, iota_bar_fp_new = update_theta_min(
                n_new, theta, theta_d_min, n_iota, m_iota, iota_bar_fp
            )
            (
                nstep_max_c_new,
                hit_rat_new,
                exist_first_ratfl_new,
                nfp_rat_new,
                nfl_rat_new,
                delta_theta_rat_new,
                n_gap_new,
            ) = update_rational(
                n_new,
                theta_d_min_new,
                n_iota_new,
                iota_bar_fp_new,
                nstep_max_c,
                hit_rat,
                exist_first_ratfl,
                nfp_rat,
                nfl_rat,
                delta_theta_rat,
                n_gap,
            )

            stop_new = jnp.where(
                (params.calc_nstep_max == 0) & (n_new == nstep_max_c_new),
                True,
                False,
            )
            stop_new = jnp.where(
                (hit_rat_new == 1) & (exist_first_ratfl_new == 0), True, stop_new
            )
            return (
                phi,
                y,
                state,
                iswst,
                bigint,
                adimax,
                aditot,
                n_new,
                theta_d_min_new,
                n_iota_new,
                m_iota_new,
                iota_bar_fp_new,
                nstep_max_c_new,
                hit_rat_new,
                exist_first_ratfl_new,
                nfp_rat_new,
                nfl_rat_new,
                delta_theta_rat_new,
                n_gap_new,
                stop_new,
            )

        return jax.lax.cond(stop, lambda c: c, do_step, carry), None

    init_carry = (
        phi,
        y,
        state,
        iswst,
        bigint,
        adimax,
        aditot,
        n,
        theta_d_min,
        n_iota,
        m_iota,
        iota_bar_fp,
        nstep_max_c,
        hit_rat,
        exist_first_ratfl,
        nfp_rat,
        nfl_rat,
        delta_theta_rat,
        n_gap,
        stop,
    )

    final_carry, _ = jax.lax.scan(scan_body, init_carry, xs=None, length=params.nstep_max)
    (
        phi,
        y,
        state,
        iswst,
        bigint,
        adimax,
        aditot,
        n,
        theta_d_min,
        n_iota,
        m_iota,
        iota_bar_fp,
        nstep_max_c,
        hit_rat,
        exist_first_ratfl,
        nfp_rat,
        nfl_rat,
        delta_theta_rat,
        n_gap,
        stop,
    ) = final_carry

    y2 = y[1]
    y3 = y[2]
    y4 = y[3]
    y3npart = y[NPQ + npart - 1]

    def rational_correction(_):
        if convergence_reset_callback is not None:
            def _reset(_):
                jax.debug.callback(convergence_reset_callback, ordered=True)
                return None

            _ = jax.lax.cond(exist_first_ratfl == 0, _reset, lambda _: None, operand=None)

        def reset_accumulators(_):
            zero_bigint = jnp.zeros_like(bigint)
            return zero_bigint, jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

        def keep_accumulators(_):
            return bigint, aditot, y2, y3, y4, y3npart

        bigint0, aditot0, y20, y30, y40, y3npart0 = jax.lax.cond(
            exist_first_ratfl == 0, reset_accumulators, keep_accumulators, operand=None
        )

        nfl_count = jnp.maximum(nfl_rat + 1 - exist_first_ratfl, 0).astype(jnp.int32)

        def nfl_cond(carry):
            idx, *_ = carry
            return idx < nfl_count

        def nfl_body(carry):
            idx, bigint_acc, aditot_acc, y2_acc, y3_acc, y4_acc, y3npart_acc = carry
            nfl = idx + exist_first_ratfl

            phi_local = phi0
            y_local = jnp.zeros(ndim)
            y_local = y_local.at[0].set(theta0 + nfl * delta_theta_rat)

            state_local = RhsState(
                isw=jnp.zeros(npart, dtype=jnp.int32),
                ipa=jnp.zeros(npart, dtype=jnp.int32),
                icount=jnp.zeros(npart, dtype=jnp.int32),
                ipmax=jnp.array(0, dtype=jnp.int32),
                pard0=jnp.array(0.0),
            )

            iswst_local = jnp.zeros(npart, dtype=jnp.int32)
            bigint_s = jnp.zeros(multra)
            adimax_s = jnp.array(0.0)
            aditot_s = jnp.array(0.0)

            def n_cond(ncarry):
                n_idx, *_ = ncarry
                return n_idx < nfp_rat

            def n_body(ncarry):
                n_idx, phi_l, y_l, state_l, iswst_l, bigint_s, adimax_s, aditot_s = ncarry

                _bmod, _gval, _geodcu, pard0, _qval = neo_eval(
                    y_l[0],
                    phi_l,
                    env.splines["b_spl"],
                    env.splines["g_spl"],
                    env.splines["k_spl"],
                    env.splines["p_spl"],
                    env.splines.get("q_spl"),
                    env.grid,
                )
                state_l = RhsState(state_l.isw, state_l.ipa, state_l.icount, state_l.ipmax, pard0)

                phi_l, y_l, state_l, iswst_l, bigint_s, adimax_s, aditot_s = integrate_period(
                    (phi_l, y_l, state_l, iswst_l, bigint_s, adimax_s, aditot_s),
                    False,
                    n_idx,
                )
                emit_convergence((nfl * nfp_rat + n_idx + 1).astype(y_l.dtype), bigint_s, y_l, aditot_s)

                return (n_idx + 1, phi_l, y_l, state_l, iswst_l, bigint_s, adimax_s, aditot_s)

            n_init = (
                jnp.array(0, dtype=jnp.int32),
                phi_local,
                y_local,
                state_local,
                iswst_local,
                bigint_s,
                adimax_s,
                aditot_s,
            )
            n_final = jax.lax.while_loop(n_cond, n_body, n_init)
            _, phi_local, y_local, state_local, iswst_local, bigint_s, adimax_s, aditot_s = n_final

            y2_s = y_local[1]
            y3_s = y_local[2]
            y4_s = y_local[3]
            y3npart_s = y_local[NPQ + npart - 1]

            return (
                idx + 1,
                bigint_acc + bigint_s,
                aditot_acc + aditot_s,
                y2_acc + y2_s,
                y3_acc + y3_s,
                y4_acc + y4_s,
                y3npart_acc + y3npart_s,
            )

        nfl_init = (jnp.array(0, dtype=jnp.int32), bigint0, aditot0, y20, y30, y40, y3npart0)
        nfl_final = jax.lax.while_loop(nfl_cond, nfl_body, nfl_init)
        _, bigint_out, aditot_out, y2_out, y3_out, y4_out, y3npart_out = nfl_final

        return bigint_out, aditot_out, y2_out, y3_out, y4_out, y3npart_out

    def rational_skip(_):
        return bigint, aditot, y2, y3, y4, y3npart

    do_rational = (hit_rat == 1) & (nfp_rat > 0)
    bigint, aditot, y2, y3, y4, y3npart = jax.lax.cond(
        do_rational, rational_correction, rational_skip, operand=None
    )

    nintfp = n

    epspar = jnp.zeros(multra)
    epstot = jnp.array(0.0)
    for m_cl in range(1, multra + 1):
        epspar = epspar.at[m_cl - 1].set(coeps * bigint[m_cl - 1] * y2 / (y3 * y3))
        epstot = epstot + epspar[m_cl - 1]

    ctrone = aditot / y2
    ctrtot = y3npart / y2
    bareph = (jnp.pi * ctrone) ** 2 / 8.0
    barept = (jnp.pi * ctrtot) ** 2 / 8.0
    drdpsi = y2 / y3
    yps = y4 * j_iota_i

    return {
        "epspar": epspar,
        "epstot": epstot,
        "ctrone": ctrone,
        "ctrtot": ctrtot,
        "bareph": bareph,
        "barept": barept,
        "drdpsi": drdpsi,
        "yps": yps,
        "y2": y2,
        "y3": y3,
        "y4": y4,
        "y3npart": y3npart,
        "bigint": bigint,
        "nintfp": nintfp,
        "hit_rat": hit_rat,
        "n_iota": n_iota,
        "m_iota": m_iota,
        "n_gap": n_gap,
        "final_n": jnp.where(hit_rat == 1, nfp_rat * (nfl_rat + 1), nintfp),
        "nfp_rat": nfp_rat,
        "nfl_rat": nfl_rat,
    }
