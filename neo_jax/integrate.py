"""Integration routines for NEO_JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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

    subsq = 1.0 - bra / env.eta
    mask = subsq > 0

    sqeta = jnp.sqrt(env.eta)
    safe_subsq = jnp.where(mask, subsq, 0.0)
    sq = jnp.sqrt(safe_subsq) * bmodm2
    p_i = jnp.where(mask, sq, 0.0)
    p_h = jnp.where(mask, sq * (4.0 / bra - 1.0 / env.eta) * geodcu / sqeta, 0.0)

    # Update particle state
    isw = jnp.where(mask, 1, jnp.where(state.isw == 1, 2, jnp.where(state.isw == 2, 2, 0)))
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
    rt0: float,
):
    """Python-loop port of flint_bo.f90 (not yet JIT-optimized)."""
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

    # Main loop
    for n in range(1, params.nstep_max + 1):
        for _j1 in range(1, params.nstep_per + 1):
            phi, y, state = rk4_step(phi, y, state, env, float(hphi))

            # Process trapped particle contributions
            p_i = y[NPQ : NPQ + npart]
            p_h = y[NPQ + npart : NPQ + 2 * npart]

            for i in range(npart):
                if int(state.isw[i]) == 2:
                    m_cl = int(state.ipa[i])
                    if m_cl < 1:
                        m_cl = 1
                    if m_cl > multra:
                        m_cl = multra
                    if int(state.ipa[i]) == 1:
                        adimax = p_i[i]
                    add_on = p_h[i] * p_h[i] / p_i[i] * iswst[i]
                    bigint = bigint.at[m_cl - 1].add(add_on)

                    iswst = iswst.at[i].set(1)
                    p_h = p_h.at[i].set(0.0)
                    p_i = p_i.at[i].set(0.0)
                    isw = state.isw.at[i].set(0)
                    icount = state.icount.at[i].set(0)
                    ipa = state.ipa.at[i].set(0)
                    state = RhsState(isw, ipa, icount, state.ipmax, state.pard0)

            y = y.at[NPQ : NPQ + npart].set(p_i)
            y = y.at[NPQ + npart : NPQ + 2 * npart].set(p_h)

            if int(state.ipmax) == 1:
                aditot = aditot + adimax
                state = RhsState(state.isw, state.ipa, state.icount, jnp.array(0, dtype=jnp.int32), state.pard0)

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

                    for i in range(npart):
                        if int(state.isw[i]) == 2:
                            m_cl = int(state.ipa[i])
                            if m_cl < 1:
                                m_cl = 1
                            if m_cl > multra:
                                m_cl = multra
                            if int(state.ipa[i]) == 1:
                                adimax_s = p_i[i]
                            add_on = p_h[i] * p_h[i] / p_i[i] * iswst[i]
                            bigint_s = bigint_s.at[m_cl - 1].add(add_on)
                            iswst = iswst.at[i].set(1)
                            p_h = p_h.at[i].set(0.0)
                            p_i = p_i.at[i].set(0.0)
                            isw = state.isw.at[i].set(0)
                            icount = state.icount.at[i].set(0)
                            ipa = state.ipa.at[i].set(0)
                            state = RhsState(isw, ipa, icount, state.ipmax, state.pard0)

                    y = y.at[NPQ : NPQ + npart].set(p_i)
                    y = y.at[NPQ + npart : NPQ + 2 * npart].set(p_h)

                    if int(state.ipmax) == 1:
                        aditot_s = aditot_s + adimax_s
                        state = RhsState(state.isw, state.ipa, state.icount, jnp.array(0, dtype=jnp.int32), state.pard0)

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

    return {
        "epspar": epspar,
        "epstot": epstot,
        "ctrone": ctrone,
        "ctrtot": ctrtot,
        "bareph": bareph,
        "barept": barept,
        "drdpsi": drdpsi,
        "yps": yps,
        "nintfp": nintfp,
        "hit_rat": hit_rat,
        "nfp_rat": nfp_rat,
        "nfl_rat": nfl_rat,
    }
