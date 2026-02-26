"""Geometry evaluation and root finding for NEO_JAX."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .splines import eva2d_fd_jax, eva2d_jax, eva2d_sd_jax, poi2d_jax

Array = jax.Array


def neo_eval(
    theta: Array,
    phi: Array,
    b_spl: Array,
    g_spl: Array,
    k_spl: Array,
    p_spl: Array,
    q_spl: Array | None,
    grid: dict,
) -> Tuple[Array, Array, Array, Array, Array]:
    """Evaluate spline fields at a point.

    Returns (bval, gval, kval, pval, qval).
    """
    ix, iy, dx, dy, ierr = poi2d_jax(
        grid["theta_int"],
        grid["phi_int"],
        grid["mt"],
        grid["mp"],
        grid["theta_start"],
        grid["theta_end"],
        grid["phi_start"],
        grid["phi_end"],
        theta,
        phi,
    )
    _ = ierr

    bval = eva2d_jax(b_spl, ix, iy, dx, dy)
    gval = eva2d_jax(g_spl, ix, iy, dx, dy)
    kval = eva2d_jax(k_spl, ix, iy, dx, dy)
    pval = eva2d_jax(p_spl, ix, iy, dx, dy)
    if q_spl is None:
        qval = jnp.array(0.0, dtype=bval.dtype)
    else:
        qval = eva2d_jax(q_spl, ix, iy, dx, dy)
    return bval, gval, kval, pval, qval


def neo_bderiv(
    theta: Array,
    phi: Array,
    b_spl: Array,
    grid: dict,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """Compute first and second derivatives of B using spline coefficients."""
    ix, iy, dx, dy, ierr = poi2d_jax(
        grid["theta_int"],
        grid["phi_int"],
        grid["mt"],
        grid["mp"],
        grid["theta_start"],
        grid["theta_end"],
        grid["phi_start"],
        grid["phi_end"],
        theta,
        phi,
    )
    _ = ierr

    fderiv = eva2d_fd_jax(b_spl, ix, iy, dx, dy)
    sderiv = eva2d_sd_jax(b_spl, ix, iy, dx, dy)

    f = fderiv[0]
    g = fderiv[1]
    dfdx = sderiv[0]
    dfdy = sderiv[1]
    dgdx = sderiv[1]
    dgdy = sderiv[2]
    return f, g, dfdx, dfdy, dgdx, dgdy


def neo_zeros2d(
    theta: Array,
    phi: Array,
    eps: float,
    iter_ma: int,
    b_spl: Array,
    grid: dict,
) -> Tuple[Array, Array, Array, Array]:
    """Newton solver for finding extrema of B.

    Returns (theta, phi, iter, error) where error=0 on success.
    """
    f, g, dfdx, dfdy, dgdx, dgdy = neo_bderiv(theta, phi, b_spl, grid)

    def cond(state):
        _, _, _, _, _, _, _, _, it, converged = state
        return jnp.logical_and(it < iter_ma, jnp.logical_not(converged))

    def body(state):
        x, y, f, g, dfdx, dfdy, dgdx, dgdy, it, _ = state
        det = dfdx * dgdy - dfdy * dgdx
        x_n = x + (dfdy * g - f * dgdy) / det
        y_n = y + (f * dgdx - dfdx * g) / det

        f_n, g_n, dfdx_n, dfdy_n, dgdx_n, dgdy_n = neo_bderiv(x_n, y_n, b_spl, grid)

        x_err = jnp.where(jnp.abs(x_n) > eps, jnp.abs((x_n - x) / x_n), jnp.abs(x_n - x))
        y_err = jnp.where(jnp.abs(y_n) > eps, jnp.abs((y_n - y) / y_n), jnp.abs(y_n - y))
        max_err = jnp.maximum(x_err, y_err)
        converged = max_err < eps

        return (x_n, y_n, f_n, g_n, dfdx_n, dfdy_n, dgdx_n, dgdy_n, it + 1, converged)

    init_state = (theta, phi, f, g, dfdx, dfdy, dgdx, dgdy, 0, False)
    theta_f, phi_f, *_rest, it, converged = jax.lax.while_loop(cond, body, init_state)
    error = jnp.where(converged, 0, 1)
    return theta_f, phi_f, it, error
