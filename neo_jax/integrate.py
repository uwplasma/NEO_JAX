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

    def tree_flatten(self):
        return (self.splines, self.grid, self.eta, self.bmod0, self.iota), None

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
    sq = jnp.where(mask, jnp.sqrt(subsq) * bmodm2, 0.0)
    p_i = sq
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
