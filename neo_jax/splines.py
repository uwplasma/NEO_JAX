"""Cubic spline utilities ported from STELLOPT NEO."""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def splreg(y: Array, h: float) -> Tuple[Array, Array, Array]:
    """Regular (non-periodic) cubic spline coefficients.

    Mirrors `splreg.f90`.
    """
    n = y.shape[0]
    ak1 = 0.0
    ak2 = 0.0
    am1 = 0.0
    am2 = 0.0

    al = jnp.zeros(n, dtype=y.dtype)
    bt = jnp.zeros(n, dtype=y.dtype)
    al = al.at[0].set(ak1)
    bt = bt.at[0].set(am1)

    c = -4.0 * h

    def forward_body(i, state):
        al, bt = state
        e = -3.0 * ((y[i + 1] - y[i]) - (y[i] - y[i - 1])) / h
        c1 = c - al[i - 1] * h
        al = al.at[i].set(h / c1)
        bt = bt.at[i].set((h * bt[i - 1] + e) / c1)
        return al, bt

    # i runs 1..n-2 inclusive
    al, bt = jax.lax.fori_loop(1, n - 1, forward_body, (al, bt))

    ci = jnp.zeros(n, dtype=y.dtype)
    if n >= 2:
        ci = ci.at[n - 1].set((am2 + ak2 * bt[n - 2]) / (1.0 - al[n - 2] * ak2))

    def backward_body(i, ci):
        # Fortran: i=1..k, i5 = n - i
        # Python index: i5 = n - i - 1
        i5 = n - i - 1
        ci = ci.at[i5].set(al[i5] * ci[i5 + 1] + bt[i5])
        return ci

    ci = jax.lax.fori_loop(1, n, backward_body, ci)

    bi = jnp.zeros(n, dtype=y.dtype)
    di = jnp.zeros(n, dtype=y.dtype)

    def coeff_body(i, state):
        bi, di = state
        bi = bi.at[i].set((y[i + 1] - y[i]) / h - h * (ci[i + 1] + 2.0 * ci[i]) / 3.0)
        di = di.at[i].set((ci[i + 1] - ci[i]) / h / 3.0)
        return bi, di

    bi, di = jax.lax.fori_loop(0, n - 1, coeff_body, (bi, di))
    return bi, ci, di


def spfper(np1: int, dtype=jnp.float64) -> Tuple[Array, Array, Array]:
    """Helper routine for periodic splines (port of spfper.f90)."""
    n = np1 - 1
    n1 = n - 1

    amx1 = jnp.zeros(np1, dtype=dtype)
    amx2 = jnp.zeros(np1, dtype=dtype)
    amx3 = jnp.zeros(np1, dtype=dtype)

    amx1 = amx1.at[0].set(2.0)
    amx2 = amx2.at[0].set(0.5)
    amx3 = amx3.at[0].set(0.5)

    if np1 > 1:
        amx1 = amx1.at[1].set(jnp.sqrt(15.0) / 2.0)
        amx2 = amx2.at[1].set(1.0 / amx1[1])
        amx3 = amx3.at[1].set(-0.25 / amx1[1])

    beta0 = 3.75

    def loop_body(i, state):
        amx1, amx2, amx3, beta = state
        beta = 4.0 - 1.0 / beta
        amx1 = amx1.at[i].set(jnp.sqrt(beta))
        amx2 = amx2.at[i].set(1.0 / amx1[i])
        amx3 = amx3.at[i].set(-amx3[i - 1] / amx1[i] / amx1[i - 1])
        return amx1, amx2, amx3, beta

    # Fortran loop i=3..n1 => Python i=2..n1-1
    if n1 > 2:
        amx1, amx2, amx3, _ = jax.lax.fori_loop(2, n1, loop_body, (amx1, amx2, amx3, beta0))
    else:
        _ = beta0

    if n1 >= 1:
        amx3 = amx3.at[n1 - 1].set(amx3[n1 - 1] + 1.0 / amx1[n1 - 1])
        amx2 = amx2.at[n1 - 1].set(amx3[n1 - 1])

    ss = jnp.sum(amx3[: n1] * amx3[: n1]) if n1 > 0 else 0.0
    if n >= 1:
        amx1 = amx1.at[n - 1].set(jnp.sqrt(4.0 - ss))

    return amx1, amx2, amx3


def splper(y: Array, h: float) -> Tuple[Array, Array, Array]:
    """Periodic cubic spline coefficients.

    Mirrors `splper.f90`.
    """
    n = y.shape[0]

    bmx = jnp.zeros(n, dtype=y.dtype)
    yl = jnp.zeros(n, dtype=y.dtype)
    amx1, amx2, amx3 = spfper(n, dtype=y.dtype)

    bmx = bmx.at[0].set(1.0e30)

    nmx = n - 1
    n1 = nmx - 1
    n2 = nmx - 2
    psi = 3.0 / (h * h)

    if nmx >= 1:
        bmx = bmx.at[nmx - 1].set((y[nmx] - 2.0 * y[nmx - 1] + y[nmx - 2]) * psi)
        bmx = bmx.at[0].set((y[1] - y[0] - y[nmx] + y[nmx - 1]) * psi)

    def bmx_body(i, bmx):
        # Fortran i=3..nmx => Python i=2..nmx-1
        bmx = bmx.at[i - 1].set((y[i] - 2.0 * y[i - 1] + y[i - 2]) * psi)
        return bmx

    if nmx > 2:
        bmx = jax.lax.fori_loop(2, nmx, bmx_body, bmx)

    if n1 >= 1:
        yl = yl.at[0].set(bmx[0] / amx1[0])

        def yl_body(i, yl):
            # Fortran i=2..n1 => Python i=1..n1-1
            yl = yl.at[i].set((bmx[i] - yl[i - 1] * amx2[i - 1]) / amx1[i])
            return yl

        if n1 > 1:
            yl = jax.lax.fori_loop(1, n1, yl_body, yl)

        ss = jnp.sum(yl[:n1] * amx3[:n1])
        yl = yl.at[nmx - 1].set((bmx[nmx - 1] - ss) / amx1[nmx - 1])

        bmx = bmx.at[nmx - 1].set(yl[nmx - 1] / amx1[nmx - 1])
        bmx = bmx.at[n1 - 1].set((yl[n1 - 1] - amx2[n1 - 1] * bmx[nmx - 1]) / amx1[n1 - 1])

        def back_body(i, bmx):
            # Fortran i=n2..1 => Python i = n2-1 .. 0
            idx = n2 - 1 - i
            bmx = bmx.at[idx].set(
                (yl[idx] - amx3[idx] * bmx[nmx - 1] - amx2[idx] * bmx[idx + 1])
                / amx1[idx]
            )
            return bmx

        if n2 >= 1:
            bmx = jax.lax.fori_loop(0, n2, back_body, bmx)

    ci = jnp.zeros(n, dtype=y.dtype)
    if nmx >= 1:
        ci = ci.at[:nmx].set(bmx[:nmx])

    bi = jnp.zeros(n, dtype=y.dtype)
    di = jnp.zeros(n, dtype=y.dtype)

    def coeff_body(i, state):
        bi, di = state
        bi = bi.at[i].set((y[i + 1] - y[i]) / h - h * (ci[i + 1] + 2.0 * ci[i]) / 3.0)
        di = di.at[i].set((ci[i + 1] - ci[i]) / h / 3.0)
        return bi, di

    if n1 >= 1:
        bi, di = jax.lax.fori_loop(0, n1, coeff_body, (bi, di))

    if nmx >= 1:
        bi = bi.at[nmx - 1].set((y[nmx] - y[nmx - 1]) / h - h * (ci[0] + 2.0 * ci[nmx - 1]) / 3.0)
        di = di.at[nmx - 1].set((ci[0] - ci[nmx - 1]) / h / 3.0)

    # Fix boundary
    bi = bi.at[n - 1].set(bi[0])
    ci = ci.at[n - 1].set(ci[0])
    di = di.at[n - 1].set(di[0])

    return bi, ci, di


def spl2d(f: Array, hx: float, hy: float, mx: int, my: int) -> Array:
    """Two-dimensional cubic spline coefficients.

    Returns array of shape (4, 4, nx, ny).
    """
    nx, ny = f.shape

    def spline_x(col):
        if mx == 0:
            bi, ci, di = splreg(col, hx)
        else:
            bi, ci, di = splper(col, hx)
        return jnp.stack([col, bi, ci, di], axis=0)

    stage1 = jax.vmap(spline_x, in_axes=1, out_axes=0)(f)  # (ny,4,nx)
    stage1 = jnp.transpose(stage1, (1, 2, 0))  # (4,nx,ny)

    def spline_y(col):
        if my == 0:
            bi, ci, di = splreg(col, hy)
        else:
            bi, ci, di = splper(col, hy)
        return jnp.stack([bi, ci, di], axis=0)  # (3, ny)

    # Flatten (k, i) axes to vmap over them.
    data = stage1.reshape((4 * nx, ny))
    stage2 = jax.vmap(spline_y, in_axes=0, out_axes=0)(data)  # (4*nx,3,ny)
    stage2 = stage2.reshape((4, nx, 3, ny))
    stage2 = jnp.transpose(stage2, (0, 2, 1, 3))  # (4,3,nx,ny)

    spl = jnp.zeros((4, 4, nx, ny), dtype=f.dtype)
    spl = spl.at[:, 0, :, :].set(stage1)
    spl = spl.at[:, 1:4, :, :].set(stage2)
    return spl


def poi2d(
    hx: float,
    hy: float,
    mx: int,
    my: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x: float,
    y: float,
):
    """Pointer calculation for spline evaluation (port of poi2d.f90).

    Returns (ix, iy, dx, dy, ierr) where ix, iy are 0-based indices.
    """
    ierr = 0

    dxx = x - xmin
    if mx == 0:
        if dxx < 0.0:
            return 0, 0, 0.0, 0.0, 1
        if x > xmax:
            return 0, 0, 0.0, 0.0, 2
    else:
        dxmax = xmax - xmin
        if dxx < 0.0:
            dxx = dxx + (1 + int(abs(dxx / dxmax))) * dxmax
        elif dxx > dxmax:
            dxx = dxx - (int(abs(dxx / dxmax))) * dxmax

    x1 = dxx / hx
    ix = int(x1)
    dx = hx * (x1 - ix)

    dyy = y - ymin
    if my == 0:
        if dyy < 0.0:
            return 0, 0, 0.0, 0.0, 3
        if y > ymax:
            return 0, 0, 0.0, 0.0, 4
    else:
        dymax = ymax - ymin
        if dyy < 0.0:
            dyy = dyy + (1 + int(abs(dyy / dymax))) * dymax
        elif dyy > dymax:
            dyy = dyy - (int(abs(dyy / dymax))) * dymax

    y1 = dyy / hy
    iy = int(y1)
    dy = hy * (y1 - iy)

    return ix, iy, dx, dy, ierr


def eva2d(spl: Array, ix: int, iy: int, dx: float, dy: float) -> float:
    """Evaluate 2D spline at given cell (port of eva2d.f90)."""
    a = []
    for l in range(4):
        a_l = spl[0, l, ix, iy] + dx * (
            spl[1, l, ix, iy] + dx * (spl[2, l, ix, iy] + dx * spl[3, l, ix, iy])
        )
        a.append(a_l)
    a = jnp.stack(a, axis=0)
    spval = a[0] + dy * (a[1] + dy * (a[2] + dy * a[3]))
    return spval


def eva2d_fd(spl: Array, ix: int, iy: int, dx: float, dy: float) -> Array:
    """Evaluate first derivatives of 2D spline (port of eva2d_fd.f90)."""
    spval = jnp.zeros((2,), dtype=spl.dtype)

    # df/dx
    for i in range(1, 4):
        muli = 1.0 if i == 1 else dx ** (i - 1)
        muli = muli * i
        for j in range(4):
            mulj = 1.0 if j == 0 else dy ** j
            spval = spval.at[0].add(spl[i, j, ix, iy] * muli * mulj)

    # df/dy
    for i in range(4):
        muli = 1.0 if i == 0 else dx ** i
        for j in range(1, 4):
            mulj = 1.0 if j == 1 else dy ** (j - 1)
            mulj = mulj * j
            spval = spval.at[1].add(spl[i, j, ix, iy] * muli * mulj)

    return spval


def eva2d_sd(spl: Array, ix: int, iy: int, dx: float, dy: float) -> Array:
    """Evaluate second derivatives of 2D spline (port of eva2d_sd.f90)."""
    spval = jnp.zeros((3,), dtype=spl.dtype)

    # d^2f/dx^2
    for i in range(2, 4):
        muli = 1.0 if i == 2 else dx ** (i - 2)
        muli = muli * (i) * (i - 1)
        for j in range(4):
            mulj = 1.0 if j == 0 else dy ** j
            spval = spval.at[0].add(spl[i, j, ix, iy] * muli * mulj)

    # d^2f/(dxdy)
    for i in range(1, 4):
        muli = 1.0 if i == 1 else dx ** (i - 1)
        muli = muli * i
        for j in range(1, 4):
            mulj = 1.0 if j == 1 else dy ** (j - 1)
            mulj = mulj * j
            spval = spval.at[1].add(spl[i, j, ix, iy] * muli * mulj)

    # d^2f/dy^2
    for i in range(4):
        muli = 1.0 if i == 0 else dx ** i
        for j in range(2, 4):
            mulj = 1.0 if j == 2 else dy ** (j - 2)
            mulj = mulj * j * (j - 1)
            spval = spval.at[2].add(spl[i, j, ix, iy] * muli * mulj)

    return spval


def poi2d_jax(
    hx: float,
    hy: float,
    mx: int,
    my: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    x: Array,
    y: Array,
):
    """JAX-friendly pointer calculation for spline evaluation."""
    ierr = jnp.int32(0)

    dxx = x - xmin
    def handle_x_nonperiodic(_):
        ierr_local = ierr
        ierr_local = jnp.where(dxx < 0.0, jnp.int32(1), ierr_local)
        ierr_local = jnp.where(x > xmax, jnp.int32(2), ierr_local)
        return dxx, ierr_local

    def handle_x_periodic(_):
        dxmax = xmax - xmin
        nwrap = jnp.floor(jnp.abs(dxx / dxmax))
        dxx_wrapped = jnp.where(dxx < 0.0, dxx + (1.0 + nwrap) * dxmax, dxx)
        dxx_wrapped = jnp.where(dxx_wrapped > dxmax, dxx_wrapped - nwrap * dxmax, dxx_wrapped)
        return dxx_wrapped, ierr

    dxx, ierr = jax.lax.cond(mx == 0, handle_x_nonperiodic, handle_x_periodic, operand=None)

    x1 = dxx / hx
    ix = jnp.floor(x1).astype(jnp.int32)
    dx = hx * (x1 - ix)

    dyy = y - ymin
    def handle_y_nonperiodic(_):
        ierr_local = ierr
        ierr_local = jnp.where(dyy < 0.0, jnp.int32(3), ierr_local)
        ierr_local = jnp.where(y > ymax, jnp.int32(4), ierr_local)
        return dyy, ierr_local

    def handle_y_periodic(_):
        dymax = ymax - ymin
        nwrap = jnp.floor(jnp.abs(dyy / dymax))
        dyy_wrapped = jnp.where(dyy < 0.0, dyy + (1.0 + nwrap) * dymax, dyy)
        dyy_wrapped = jnp.where(dyy_wrapped > dymax, dyy_wrapped - nwrap * dymax, dyy_wrapped)
        return dyy_wrapped, ierr

    dyy, ierr = jax.lax.cond(my == 0, handle_y_nonperiodic, handle_y_periodic, operand=None)

    y1 = dyy / hy
    iy = jnp.floor(y1).astype(jnp.int32)
    dy = hy * (y1 - iy)

    return ix, iy, dx, dy, ierr


def eva2d_jax(spl: Array, ix: Array, iy: Array, dx: Array, dy: Array) -> Array:
    """JAX-friendly spline evaluation."""
    coeff = jnp.take(spl, ix, axis=2)
    coeff = jnp.take(coeff, iy, axis=2)
    a = coeff[0, :] + dx * (coeff[1, :] + dx * (coeff[2, :] + dx * coeff[3, :]))
    spval = a[0] + dy * (a[1] + dy * (a[2] + dy * a[3]))
    return spval


def eva2d_fd_jax(spl: Array, ix: Array, iy: Array, dx: Array, dy: Array) -> Array:
    """JAX-friendly first derivatives of spline."""
    coeff = jnp.take(spl, ix, axis=2)
    coeff = jnp.take(coeff, iy, axis=2)

    # df/dx
    sp0 = 0.0
    for i in range(1, 4):
        muli = (1.0 if i == 1 else dx ** (i - 1)) * i
        for j in range(4):
            mulj = 1.0 if j == 0 else dy ** j
            sp0 = sp0 + coeff[i, j] * muli * mulj

    # df/dy
    sp1 = 0.0
    for i in range(4):
        muli = 1.0 if i == 0 else dx ** i
        for j in range(1, 4):
            mulj = (1.0 if j == 1 else dy ** (j - 1)) * j
            sp1 = sp1 + coeff[i, j] * muli * mulj

    return jnp.array([sp0, sp1], dtype=spl.dtype)


def eva2d_sd_jax(spl: Array, ix: Array, iy: Array, dx: Array, dy: Array) -> Array:
    """JAX-friendly second derivatives of spline."""
    coeff = jnp.take(spl, ix, axis=2)
    coeff = jnp.take(coeff, iy, axis=2)

    sp0 = 0.0
    for i in range(2, 4):
        muli = (1.0 if i == 2 else dx ** (i - 2)) * i * (i - 1)
        for j in range(4):
            mulj = 1.0 if j == 0 else dy ** j
            sp0 = sp0 + coeff[i, j] * muli * mulj

    sp1 = 0.0
    for i in range(1, 4):
        muli = (1.0 if i == 1 else dx ** (i - 1)) * i
        for j in range(1, 4):
            mulj = (1.0 if j == 1 else dy ** (j - 1)) * j
            sp1 = sp1 + coeff[i, j] * muli * mulj

    sp2 = 0.0
    for i in range(4):
        muli = 1.0 if i == 0 else dx ** i
        for j in range(2, 4):
            mulj = (1.0 if j == 2 else dy ** (j - 2)) * j * (j - 1)
            sp2 = sp2 + coeff[i, j] * muli * mulj

    return jnp.array([sp0, sp1, sp2], dtype=spl.dtype)
