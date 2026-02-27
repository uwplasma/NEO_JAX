"""Fourier summations and derived quantities for Boozer geometry."""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

Array = jax.Array


def _apply_periodic_edges(arr: Array) -> Array:
    # Enforce periodic boundaries by copying first row/column.
    arr = arr.at[-1, :].set(arr[0, :])
    arr = arr.at[:, -1].set(arr[:, 0])
    return arr


def fourier_sums(
    theta_arr: Array,
    phi_arr: Array,
    rmnc: Array,
    zmns: Array,
    lmns: Array,
    bmnc: Array,
    ixm: Array,
    ixn: Array,
    nfp: int,
    max_m_mode: int,
    max_n_mode: int,
    lasym: bool = False,
    rmns: Array | None = None,
    zmnc: Array | None = None,
    lmnc: Array | None = None,
    bmns: Array | None = None,
) -> Dict[str, Array]:
    """Compute Fourier sums for a single flux surface.

    This mirrors `neo_fourier.f90` but uses direct trig evaluation.
    """
    m = ixm.astype(theta_arr.dtype)
    n = ixn.astype(theta_arr.dtype)

    mask = (jnp.abs(m) <= max_m_mode) & (jnp.abs(n) <= max_n_mode)
    m = m[mask]
    n = n[mask]
    rmnc = rmnc[mask]
    zmns = zmns[mask]
    lmns = lmns[mask]
    bmnc = bmnc[mask]

    theta = theta_arr[:, None]
    phi = phi_arr[:, None]

    # Match Fortran's trig combination to reduce rounding drift:
    # cos(m*theta - n*phi) = cos(m*theta)*cos(n*phi) + sin(m*theta)*sin(n*phi)
    # sin(m*theta - n*phi) = sin(m*theta)*cos(n*phi) - cos(m*theta)*sin(n*phi)
    cos_mth = jnp.cos(theta * m[None, :])
    sin_mth = jnp.sin(theta * m[None, :])
    cos_nph = jnp.cos(phi * n[None, :])
    sin_nph = jnp.sin(phi * n[None, :])

    cosv = cos_mth[:, None, :] * cos_nph[None, :, :] + sin_mth[:, None, :] * sin_nph[None, :, :]
    sinv = sin_mth[:, None, :] * cos_nph[None, :, :] - cos_mth[:, None, :] * sin_nph[None, :, :]

    r = jnp.sum(rmnc[None, None, :] * cosv, axis=2)
    z = jnp.sum(zmns[None, None, :] * sinv, axis=2)
    l = jnp.sum(lmns[None, None, :] * sinv, axis=2)
    b = jnp.sum(bmnc[None, None, :] * cosv, axis=2)

    r_tb = jnp.sum(-m[None, None, :] * rmnc[None, None, :] * sinv, axis=2)
    r_pb = jnp.sum(n[None, None, :] * rmnc[None, None, :] * sinv, axis=2)
    z_tb = jnp.sum(m[None, None, :] * zmns[None, None, :] * cosv, axis=2)
    z_pb = jnp.sum(-n[None, None, :] * zmns[None, None, :] * cosv, axis=2)
    p_tb = jnp.sum(-m[None, None, :] * lmns[None, None, :] * cosv, axis=2)
    p_pb = jnp.sum(n[None, None, :] * lmns[None, None, :] * cosv, axis=2)
    b_tb = jnp.sum(-m[None, None, :] * bmnc[None, None, :] * sinv, axis=2)
    b_pb = jnp.sum(n[None, None, :] * bmnc[None, None, :] * sinv, axis=2)

    if lasym:
        if rmns is None or zmnc is None or lmnc is None or bmns is None:
            raise ValueError("Asymmetric terms requested but coefficients missing")
        rmns = rmns[mask]
        zmnc = zmnc[mask]
        lmnc = lmnc[mask]
        bmns = bmns[mask]

        r = r + jnp.sum(rmns[None, None, :] * sinv, axis=2)
        z = z + jnp.sum(zmnc[None, None, :] * cosv, axis=2)
        l = l + jnp.sum(lmnc[None, None, :] * cosv, axis=2)
        b = b + jnp.sum(bmns[None, None, :] * sinv, axis=2)

        r_tb = r_tb + jnp.sum(m[None, None, :] * rmns[None, None, :] * cosv, axis=2)
        r_pb = r_pb + jnp.sum(-n[None, None, :] * rmns[None, None, :] * cosv, axis=2)
        z_tb = z_tb + jnp.sum(-m[None, None, :] * zmnc[None, None, :] * sinv, axis=2)
        z_pb = z_pb + jnp.sum(n[None, None, :] * zmnc[None, None, :] * sinv, axis=2)
        p_tb = p_tb + jnp.sum(m[None, None, :] * lmnc[None, None, :] * sinv, axis=2)
        p_pb = p_pb + jnp.sum(-n[None, None, :] * lmnc[None, None, :] * sinv, axis=2)
        b_tb = b_tb + jnp.sum(m[None, None, :] * bmns[None, None, :] * cosv, axis=2)
        b_pb = b_pb + jnp.sum(-n[None, None, :] * bmns[None, None, :] * cosv, axis=2)

    # Convert lambda derivatives to Boozer phi derivatives
    twopi = 2.0 * jnp.pi
    p_tb = p_tb * twopi / nfp
    p_pb = 1.0 + p_pb * twopi / nfp

    # Enforce periodicity
    r = _apply_periodic_edges(r)
    z = _apply_periodic_edges(z)
    l = _apply_periodic_edges(l)
    b = _apply_periodic_edges(b)
    r_tb = _apply_periodic_edges(r_tb)
    r_pb = _apply_periodic_edges(r_pb)
    z_tb = _apply_periodic_edges(z_tb)
    z_pb = _apply_periodic_edges(z_pb)
    p_tb = _apply_periodic_edges(p_tb)
    p_pb = _apply_periodic_edges(p_pb)
    b_tb = _apply_periodic_edges(b_tb)
    b_pb = _apply_periodic_edges(b_pb)

    return {
        "r": r,
        "z": z,
        "l": l,
        "b": b,
        "r_tb": r_tb,
        "r_pb": r_pb,
        "z_tb": z_tb,
        "z_pb": z_pb,
        "p_tb": p_tb,
        "p_pb": p_pb,
        "b_tb": b_tb,
        "b_pb": b_pb,
    }


def derived_quantities(
    fourier: Dict[str, Array],
    curr_pol: Array,
    curr_tor: Array,
    iota: Array,
) -> Dict[str, Array]:
    """Compute derived quantities from Fourier sums (neo_fourier.f90)."""
    r = fourier["r"]
    b = fourier["b"]
    r_tb = fourier["r_tb"]
    r_pb = fourier["r_pb"]
    z_tb = fourier["z_tb"]
    z_pb = fourier["z_pb"]
    p_tb = fourier["p_tb"]
    p_pb = fourier["p_pb"]
    b_tb = fourier["b_tb"]
    b_pb = fourier["b_pb"]

    gtbtb = r_tb * r_tb + z_tb * z_tb + r * r * p_tb * p_tb
    gpbpb = r_pb * r_pb + z_pb * z_pb + r * r * p_pb * p_pb
    gtbpb = r_tb * r_pb + z_tb * z_pb + r * r * p_tb * p_pb

    fac = curr_pol + iota * curr_tor
    isqrg = b * b / fac
    sqrg11 = jnp.sqrt(gtbtb * gpbpb - gtbpb * gtbpb) * isqrg

    kg = (curr_tor * b_pb - curr_pol * b_tb) / fac
    pard = b_pb + iota * b_tb

    return {
        "gtbtb": gtbtb,
        "gpbpb": gpbpb,
        "gtbpb": gtbpb,
        "isqrg": isqrg,
        "sqrg11": sqrg11,
        "kg": kg,
        "pard": pard,
    }
