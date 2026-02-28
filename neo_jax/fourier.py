"""Fourier summations and derived quantities for Boozer geometry."""

from __future__ import annotations

from typing import Dict, Tuple
import os

import jax
import jax.numpy as jnp

Array = jax.Array


def _apply_periodic_edges(arr: Array) -> Array:
    # Enforce periodic boundaries by copying first row/column.
    arr = arr.at[-1, :].set(arr[0, :])
    arr = arr.at[:, -1].set(arr[:, 0])
    return arr


def _fourier_sums_vectorized(
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
    *,
    skip_mask: bool = False,
    lasym: bool = False,
    rmns: Array | None = None,
    zmnc: Array | None = None,
    lmnc: Array | None = None,
    bmns: Array | None = None,
) -> Dict[str, Array]:
    """Vectorized Fourier sums (fast but allocates theta×phi×modes)."""
    m = ixm.astype(theta_arr.dtype)
    n = ixn.astype(theta_arr.dtype)

    if skip_mask:
        mask = None
    else:
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
        if mask is not None:
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


def _fourier_sums_streamed(
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
    *,
    skip_mask: bool = False,
    lasym: bool = False,
    rmns: Array | None = None,
    zmnc: Array | None = None,
    lmnc: Array | None = None,
    bmns: Array | None = None,
) -> Dict[str, Array]:
    """Streamed Fourier sums to avoid theta×phi×mode temporaries."""
    m = ixm.astype(theta_arr.dtype)
    n = ixn.astype(theta_arr.dtype)

    if skip_mask:
        mask = None
    else:
        mask = (jnp.abs(m) <= max_m_mode) & (jnp.abs(n) <= max_n_mode)
        m = m[mask]
        n = n[mask]
        rmnc = rmnc[mask]
        zmns = zmns[mask]
        lmns = lmns[mask]
        bmnc = bmnc[mask]

    theta = theta_arr
    phi = phi_arr

    cos_mth = jnp.cos(theta[:, None] * m[None, :])
    sin_mth = jnp.sin(theta[:, None] * m[None, :])
    cos_nph = jnp.cos(phi[:, None] * n[None, :])
    sin_nph = jnp.sin(phi[:, None] * n[None, :])

    theta_n = theta.shape[0]
    phi_n = phi.shape[0]

    def zero_field():
        return jnp.zeros((theta_n, phi_n), dtype=theta_arr.dtype)

    carry = (
        zero_field(),  # r
        zero_field(),  # z
        zero_field(),  # l
        zero_field(),  # b
        zero_field(),  # r_tb
        zero_field(),  # r_pb
        zero_field(),  # z_tb
        zero_field(),  # z_pb
        zero_field(),  # p_tb
        zero_field(),  # p_pb
        zero_field(),  # b_tb
        zero_field(),  # b_pb
    )

    def body(k, state):
        r, z, l, b, r_tb, r_pb, z_tb, z_pb, p_tb, p_pb, b_tb, b_pb = state
        cosv = cos_mth[:, k][:, None] * cos_nph[:, k][None, :] + sin_mth[:, k][:, None] * sin_nph[:, k][None, :]
        sinv = sin_mth[:, k][:, None] * cos_nph[:, k][None, :] - cos_mth[:, k][:, None] * sin_nph[:, k][None, :]

        r = r + rmnc[k] * cosv
        z = z + zmns[k] * sinv
        l = l + lmns[k] * sinv
        b = b + bmnc[k] * cosv

        r_tb = r_tb - m[k] * rmnc[k] * sinv
        r_pb = r_pb + n[k] * rmnc[k] * sinv
        z_tb = z_tb + m[k] * zmns[k] * cosv
        z_pb = z_pb - n[k] * zmns[k] * cosv
        p_tb = p_tb - m[k] * lmns[k] * cosv
        p_pb = p_pb + n[k] * lmns[k] * cosv
        b_tb = b_tb - m[k] * bmnc[k] * sinv
        b_pb = b_pb + n[k] * bmnc[k] * sinv

        return (r, z, l, b, r_tb, r_pb, z_tb, z_pb, p_tb, p_pb, b_tb, b_pb)

    r, z, l, b, r_tb, r_pb, z_tb, z_pb, p_tb, p_pb, b_tb, b_pb = jax.lax.fori_loop(
        0, m.shape[0], body, carry
    )

    if lasym:
        if rmns is None or zmnc is None or lmnc is None or bmns is None:
            raise ValueError("Asymmetric terms requested but coefficients missing")
        if mask is not None:
            rmns = rmns[mask]
            zmnc = zmnc[mask]
            lmnc = lmnc[mask]
            bmns = bmns[mask]

        def body_asym(k, state):
            r, z, l, b, r_tb, r_pb, z_tb, z_pb, p_tb, p_pb, b_tb, b_pb = state
            cosv = cos_mth[:, k][:, None] * cos_nph[:, k][None, :] + sin_mth[:, k][:, None] * sin_nph[:, k][None, :]
            sinv = sin_mth[:, k][:, None] * cos_nph[:, k][None, :] - cos_mth[:, k][:, None] * sin_nph[:, k][None, :]

            r = r + rmns[k] * sinv
            z = z + zmnc[k] * cosv
            l = l + lmnc[k] * cosv
            b = b + bmns[k] * sinv

            r_tb = r_tb + m[k] * rmns[k] * cosv
            r_pb = r_pb - n[k] * rmns[k] * cosv
            z_tb = z_tb - m[k] * zmnc[k] * sinv
            z_pb = z_pb + n[k] * zmnc[k] * sinv
            p_tb = p_tb + m[k] * lmnc[k] * sinv
            p_pb = p_pb - n[k] * lmnc[k] * sinv
            b_tb = b_tb + m[k] * bmns[k] * cosv
            b_pb = b_pb - n[k] * bmns[k] * cosv

            return (r, z, l, b, r_tb, r_pb, z_tb, z_pb, p_tb, p_pb, b_tb, b_pb)

        r, z, l, b, r_tb, r_pb, z_tb, z_pb, p_tb, p_pb, b_tb, b_pb = jax.lax.fori_loop(
            0, m.shape[0], body_asym, (r, z, l, b, r_tb, r_pb, z_tb, z_pb, p_tb, p_pb, b_tb, b_pb)
        )

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
    *,
    skip_mask: bool = False,
    lasym: bool = False,
    rmns: Array | None = None,
    zmnc: Array | None = None,
    lmnc: Array | None = None,
    bmns: Array | None = None,
) -> Dict[str, Array]:
    """Compute Fourier sums for a single flux surface.

    This mirrors `neo_fourier.f90` but uses direct trig evaluation.
    Set `NEO_JAX_FOURIER_MODE=streamed` to reduce memory by avoiding
    theta×phi×mode temporaries.
    """
    mode = os.getenv("NEO_JAX_FOURIER_MODE", "vectorized").strip().lower()
    if mode == "streamed":
        return _fourier_sums_streamed(
            theta_arr,
            phi_arr,
            rmnc,
            zmns,
            lmns,
            bmnc,
            ixm,
            ixn,
            nfp,
            max_m_mode,
            max_n_mode,
            skip_mask=skip_mask,
            lasym=lasym,
            rmns=rmns,
            zmnc=zmnc,
            lmnc=lmnc,
            bmns=bmns,
        )
    if mode == "vectorized":
        return _fourier_sums_vectorized(
            theta_arr,
            phi_arr,
            rmnc,
            zmns,
            lmns,
            bmnc,
            ixm,
            ixn,
            nfp,
            max_m_mode,
            max_n_mode,
            skip_mask=skip_mask,
            lasym=lasym,
            rmns=rmns,
            zmnc=zmnc,
            lmnc=lmnc,
            bmns=bmns,
        )
    raise ValueError(f"Unsupported NEO_JAX_FOURIER_MODE '{mode}'")


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
