"""Parallel-current solve for the legacy ``calc_cur = 1`` NEO path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import jax
import jax.numpy as jnp

from .geometry import neo_eval


NPQ_CUR = 8
NINTFP_CUR = 10000


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CurrentParams:
    npart_cur: int
    alpha_cur: float
    nstep_per: int
    nfp: int
    write_cur_inte: bool = False

    def tree_flatten(self):
        return (
            self.npart_cur,
            self.alpha_cur,
            self.nstep_per,
            self.nfp,
            self.write_cur_inte,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CurrentEnv:
    splines: dict
    grid: dict
    iota: jax.Array
    fac: jax.Array
    bmod0: jax.Array
    theta0: jax.Array
    phi0: jax.Array
    t: jax.Array
    y_part: jax.Array
    ht: jax.Array

    def tree_flatten(self):
        return (
            self.splines,
            self.grid,
            self.iota,
            self.fac,
            self.bmod0,
            self.theta0,
            self.phi0,
            self.t,
            self.y_part,
            self.ht,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def _sample_index(npart_cur: int, idx: int) -> int:
    return min(max(idx, 0), npart_cur - 1)


def _current_env(surface, env, *, npart_cur: int, alpha_cur: float, nfp: int) -> CurrentEnv:
    tmin = jnp.asarray(1.0e-3, dtype=surface.bmref.dtype)
    tmax = jnp.asarray(1.0, dtype=surface.bmref.dtype)
    ht = (tmax - tmin) / jnp.asarray(max(npart_cur - 1, 1), dtype=surface.bmref.dtype)
    t = tmin + ht * jnp.arange(npart_cur, dtype=surface.bmref.dtype)
    y_part = 1.0 - t ** jnp.asarray(alpha_cur, dtype=surface.bmref.dtype)
    fac = jnp.asarray(env.curr_pol + env.iota * env.curr_tor, dtype=surface.bmref.dtype)
    return CurrentEnv(
        splines=env.splines,
        grid=env.grid,
        iota=jnp.asarray(env.iota, dtype=surface.bmref.dtype),
        fac=fac,
        bmod0=jnp.asarray(surface.bmref, dtype=surface.bmref.dtype),
        theta0=jnp.asarray(surface.theta_bmax, dtype=surface.bmref.dtype),
        phi0=jnp.asarray(surface.phi_bmax, dtype=surface.bmref.dtype),
        t=t,
        y_part=y_part,
        ht=ht,
    )


def rhs_cur(phi: jax.Array, y: jax.Array, env: CurrentEnv) -> jax.Array:
    theta = y[0]
    bmod, gval, geodcu, pardeb, qval = neo_eval(
        theta,
        phi,
        env.splines["b_spl"],
        env.splines["g_spl"],
        env.splines["k_spl"],
        env.splines["p_spl"],
        env.splines.get("q_spl"),
        env.grid,
    )

    bra = bmod / env.bmod0
    bmodm2 = 1.0 / (bra * bra)
    bmodm3 = bmodm2 / bra
    curfac = geodcu * env.fac / env.bmod0

    yfac = 1.0 - env.y_part * bra
    yfac_safe = jnp.where(yfac <= 0.0, jnp.asarray(1.0e-30, dtype=y.dtype), yfac)
    sqyfac = jnp.sqrt(jnp.maximum(yfac, 0.0))

    dery = jnp.zeros_like(y)
    dery = dery.at[0].set(env.iota)
    dery = dery.at[1].set(bmodm2)
    dery = dery.at[2].set(bmodm2 * gval)
    dery = dery.at[3].set(curfac * bmodm3)
    dery = dery.at[4].set(qval * bmodm2 * dery[3])
    dery = dery.at[5].set(qval * bmodm2)
    dery = dery.at[6].set(dery[3])
    dery = dery.at[7].set(1.0)

    p_k1 = y[NPQ_CUR + env.t.shape[0] : NPQ_CUR + 2 * env.t.shape[0]]
    pd_l = bmodm2 * sqyfac
    k_fac1 = -(2.0 / bra + env.y_part / (2.0 * yfac_safe)) * pardeb / env.bmod0
    k_fac2 = bmodm2 / yfac_safe * curfac
    pd_k1 = k_fac1 * p_k1 + k_fac2
    pd_k = p_k1

    dery = dery.at[NPQ_CUR : NPQ_CUR + env.t.shape[0]].set(pd_l)
    dery = dery.at[NPQ_CUR + env.t.shape[0] : NPQ_CUR + 2 * env.t.shape[0]].set(pd_k1)
    dery = dery.at[NPQ_CUR + 2 * env.t.shape[0] : NPQ_CUR + 3 * env.t.shape[0]].set(pd_k)
    return dery


def rk4_cur_step(phi: jax.Array, y: jax.Array, h: jax.Array, env: CurrentEnv) -> tuple[jax.Array, jax.Array]:
    hh = h / 2.0
    h6 = h / 6.0

    k1 = rhs_cur(phi, y, env)
    y1 = y + hh * k1

    k2 = rhs_cur(phi + hh, y1, env)
    y2 = y + hh * k2

    k3 = rhs_cur(phi + hh, y2, env)
    y3 = y + h * k3

    k4 = rhs_cur(phi + h, y3, env)

    return phi + h, y + h6 * (k1 + k4 + 2.0 * (k2 + k3))


def current_metrics(y: jax.Array, env: CurrentEnv, params: CurrentParams) -> Dict[str, jax.Array]:
    npart_cur = params.npart_cur
    p_bm2 = y[1]
    p_bm2gv = y[2]
    p_lamps = y[3]
    p_lamb1n = y[4]
    p_lamb1d = y[5]
    p_lamb2n = y[6]
    p_lamb2d = y[7]
    p_l = y[NPQ_CUR : NPQ_CUR + npart_cur]
    p_k1 = y[NPQ_CUR + npart_cur : NPQ_CUR + 2 * npart_cur]
    p_k = y[NPQ_CUR + 2 * npart_cur : NPQ_CUR + 3 * npart_cur]

    avnabpsi = p_bm2gv / p_bm2
    lambda_b = jnp.sum(env.t ** (params.alpha_cur - 1.0) * env.y_part * env.y_part * p_k / p_l)
    lambda_b = -3.0 / 8.0 * lambda_b * params.alpha_cur * env.ht
    lambda_ps1 = 2.0 * p_lamb1n / p_lamb1d
    lambda_ps2 = 2.0 * p_lamb2n / p_lamb2d
    lambda_b1 = lambda_b + lambda_ps1
    lambda_b2 = lambda_b + lambda_ps2

    i1 = _sample_index(npart_cur, 0)
    i50 = _sample_index(npart_cur, 49)
    i100 = _sample_index(npart_cur, 99)
    return {
        "avnabpsi": avnabpsi,
        "lambda_b": lambda_b,
        "lambda_ps1": lambda_ps1,
        "lambda_ps2": lambda_ps2,
        "lambda_b1": lambda_b1,
        "lambda_b2": lambda_b2,
        "p_lamps": p_lamps,
        "p_lamb1n": p_lamb1n,
        "p_lamb1d": p_lamb1d,
        "p_lamb2n": p_lamb2n,
        "p_lamb2d": p_lamb2d,
        "pk_over_l_1": p_k[i1] / p_l[i1],
        "p_l_1": p_l[i1],
        "p_k_1": p_k[i1],
        "p_k1_1": p_k1[i1],
        "pk_over_l_50": p_k[i50] / p_l[i50],
        "p_l_50": p_l[i50],
        "p_k_50": p_k[i50],
        "p_k1_50": p_k1[i50],
        "pk_over_l_100": p_k[i100] / p_l[i100],
        "p_l_100": p_l[i100],
        "p_k_100": p_k[i100],
        "p_k1_100": p_k1[i100],
    }


def _history_row(n: jax.Array, metrics: Dict[str, jax.Array]) -> jax.Array:
    return jnp.asarray(
        [
            n,
            metrics["avnabpsi"],
            metrics["lambda_b"],
            metrics["lambda_ps1"],
            metrics["lambda_ps2"],
            metrics["lambda_b1"],
            metrics["lambda_b2"],
            metrics["p_lamps"],
            metrics["p_lamb1n"],
            metrics["p_lamb1d"],
            metrics["p_lamb2n"],
            metrics["p_lamb2d"],
            metrics["pk_over_l_1"],
            metrics["p_l_1"],
            metrics["p_k_1"],
            metrics["p_k1_1"],
            metrics["pk_over_l_50"],
            metrics["p_l_50"],
            metrics["p_k_50"],
            metrics["p_k1_50"],
            metrics["pk_over_l_100"],
            metrics["p_l_100"],
            metrics["p_k_100"],
            metrics["p_k1_100"],
        ]
    )


def flint_cur_jax(surface, params: CurrentParams, env) -> Dict[str, jax.Array]:
    cur_env = _current_env(surface, env, npart_cur=params.npart_cur, alpha_cur=params.alpha_cur, nfp=params.nfp)
    ndim = NPQ_CUR + 3 * params.npart_cur
    hphi = 2.0 * jnp.pi / (params.nstep_per * params.nfp)
    y0 = jnp.zeros(ndim, dtype=cur_env.theta0.dtype).at[0].set(cur_env.theta0)
    phi0 = cur_env.phi0

    def period_step(carry, _):
        phi, y = carry

        def inner(inner_carry, _):
            phi_inner, y_inner = inner_carry
            return rk4_cur_step(phi_inner, y_inner, hphi, cur_env), None

        (phi_out, y_out), _ = jax.lax.scan(inner, (phi, y), xs=None, length=params.nstep_per)
        metrics = current_metrics(y_out, cur_env, params)
        row = _history_row(0.0, metrics)
        return (phi_out, y_out), row

    def outer(carry, n_idx):
        phi, y = carry

        def inner(inner_carry, _):
            phi_inner, y_inner = inner_carry
            return rk4_cur_step(phi_inner, y_inner, hphi, cur_env), None

        (phi_out, y_out), _ = jax.lax.scan(inner, (phi, y), xs=None, length=params.nstep_per)
        metrics = current_metrics(y_out, cur_env, params)
        row = _history_row(n_idx + 1, metrics)
        return (phi_out, y_out), row

    (phi_final, y_final), history_rows = jax.lax.scan(
        outer, (phi0, y0), xs=jnp.arange(NINTFP_CUR, dtype=cur_env.theta0.dtype)
    )
    metrics = current_metrics(y_final, cur_env, params)

    avnabpsi = metrics["avnabpsi"]
    return {
        "avnabpsi": avnabpsi,
        "lambda_b": metrics["lambda_b"] / avnabpsi,
        "lambda_ps1": metrics["lambda_ps1"] / avnabpsi,
        "lambda_ps2": metrics["lambda_ps2"] / avnabpsi,
        "lambda_b1": metrics["lambda_b1"] / avnabpsi,
        "lambda_b2": metrics["lambda_b2"] / avnabpsi,
        "history_rows": history_rows if params.write_cur_inte else None,
    }
