"""Geometry evaluation and root finding for NEO_JAX."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp
from solvax import piecewise_bicubic_evaluate

Array = jax.Array


def _evaluate(spline: Array, theta: Array, phi: Array, grid: dict, derivative):
    """Evaluate the surface spline of :func:`neo_jax.surface.build_splines` on its grid.

    ``grid["mt"]`` and ``grid["mp"]`` may be traced, so a periodic axis reduces its
    points here instead of through the static ``periodic`` flag of SOLVAX."""
    knots, points = [], []
    for axis, flag, n, p in (("theta", "mt", spline.shape[1], theta),
                             ("phi", "mp", spline.shape[3], phi)):
        start, step = grid[f"{axis}_start"], grid[f"{axis}_int"]
        points.append(jnp.where(grid[flag] == 0, p, start + jnp.mod(p - start, step * n)))
        knots.append(start + step * jnp.arange(n + 1))
    return piecewise_bicubic_evaluate(*knots, spline, *points, derivative=derivative,
                                      uniform=(True, True))


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
    bval, gval, kval, pval = (_evaluate(s, theta, phi, grid, (0, 0))
                              for s in (b_spl, g_spl, k_spl, p_spl))
    if q_spl is None:
        qval = jnp.array(0.0, dtype=bval.dtype)
    else:
        qval = _evaluate(q_spl, theta, phi, grid, (0, 0))
    return bval, gval, kval, pval, qval


def neo_bderiv(
    theta: Array,
    phi: Array,
    b_spl: Array,
    grid: dict,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
    """First and second derivatives of B: (f, g, df/dx, df/dy, dg/dx, dg/dy) with f = dB/dtheta
    and g = dB/dphi."""
    f, g, dfdx, dfdy, dgdy = _evaluate(b_spl, theta, phi, grid,
                                       ((1, 0), (0, 1), (2, 0), (1, 1), (0, 2)))
    return f, g, dfdx, dfdy, dfdy, dgdy


def neo_zeros2d(
    theta: Array,
    phi: Array,
    eps: float,
    iter_ma: int,
    b_spl: Array,
    grid: dict,
    *, phi_invariant: bool = False,
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
        stationary = (f == 0) & (g == 0)
        det = jnp.where(phi_invariant | stationary, 1.0, dfdx * dgdy - dfdy * dgdx)
        dx = jnp.where(phi_invariant, -f / jnp.where(dfdx == 0, 1.0, dfdx),
                       (dfdy * g - f * dgdy) / det)
        x_n = x + jnp.where(stationary, 0.0, dx)
        y_n = y + jnp.where(phi_invariant | stationary, 0.0, (f * dgdx - dfdx * g) / det)

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
