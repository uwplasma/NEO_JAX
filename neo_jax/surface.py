"""Surface initialization and spline construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import math
import numpy as np

import jax
import jax.numpy as jnp

from .fourier import derived_quantities, fourier_sums
from .geometry import neo_zeros2d
from .splines import spl2d

Array = jax.Array


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SurfaceData:
    b_min: Array
    b_max: Array
    theta_bmin: Array
    phi_bmin: Array
    theta_bmax: Array
    phi_bmax: Array
    bmref: Array
    fields: Dict[str, Array]
    splines: Dict[str, Array]

    def tree_flatten(self):
        children = (self.b_min, self.b_max, self.theta_bmin, self.phi_bmin,
                    self.theta_bmax, self.phi_bmax, self.bmref, self.fields, self.splines)
        return children, None

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(*children)


def build_splines(
    fields: Dict[str, Array],
    theta_int: float,
    phi_int: float,
    mt: int,
    mp: int,
    calc_cur: bool = False,
) -> Dict[str, Array]:
    b_spl = spl2d(fields["b"], theta_int, phi_int, mt, mp)
    g_spl = spl2d(fields["sqrg11"], theta_int, phi_int, mt, mp)
    k_spl = spl2d(fields["kg"], theta_int, phi_int, mt, mp)
    p_spl = spl2d(fields["pard"], theta_int, phi_int, mt, mp)
    splines = {"b_spl": b_spl, "g_spl": g_spl, "k_spl": k_spl, "p_spl": p_spl}
    if calc_cur and "bqtphi" in fields:
        splines["q_spl"] = spl2d(fields["bqtphi"], theta_int, phi_int, mt, mp)
    return splines


def _select_extremum_index(
    b: Array,
    theta_arr: Array,
    phi_arr: Array,
    ixm: Array,
    ixn: Array,
    bmnc: Array,
    max_m_mode: int,
    max_n_mode: int,
    *,
    find_max: bool,
) -> Tuple[Array, Array]:
    """Select extremum index with a Fortran-like tie-breaker."""
    b_np = np.asarray(b)
    if find_max:
        extremum = b_np.max()
        tol = 1.0e-12 * max(1.0, abs(extremum))
        mask = b_np >= (extremum - tol)
    else:
        extremum = b_np.min()
        tol = 1.0e-12 * max(1.0, abs(extremum))
        mask = b_np <= (extremum + tol)

    # Candidate indices in Fortran order (theta index varies fastest).
    mask_t = mask.T
    flat = mask_t.reshape(-1)
    candidates = np.nonzero(flat)[0]
    if candidates.size == 0:
        idx = int(np.argmax(flat)) if find_max else int(np.argmin(flat))
        j, i = np.unravel_index(idx, mask_t.shape)
        return jnp.asarray(i), jnp.asarray(j)

    theta_np = np.asarray(theta_arr)
    phi_np = np.asarray(phi_arr)
    ixm_np = np.asarray(ixm)
    ixn_np = np.asarray(ixn)
    bmnc_np = np.asarray(bmnc)

    def b_at(theta: float, phi: float) -> float:
        total = 0.0
        for m, n, coeff in zip(ixm_np, ixn_np, bmnc_np):
            if abs(m) <= max_m_mode and abs(n) <= max_n_mode:
                total += float(coeff) * math.cos(float(m) * theta - float(n) * phi)
        return total

    best_val = None
    best_i = None
    best_j = None
    for idx in candidates:
        j, i = np.unravel_index(int(idx), mask_t.shape)
        theta = float(theta_np[i])
        phi = float(phi_np[j])
        val = b_at(theta, phi)
        if best_val is None:
            best_val = val
            best_i, best_j = i, j
            continue
        if find_max:
            if val > best_val:
                best_val = val
                best_i, best_j = i, j
        else:
            if val < best_val:
                best_val = val
                best_i, best_j = i, j

    return jnp.asarray(best_i), jnp.asarray(best_j)


def init_surface(
    theta_arr: Array,
    phi_arr: Array,
    coeffs: Dict[str, Array],
    ixm: Array,
    ixn: Array,
    nfp: int,
    max_m_mode: int,
    max_n_mode: int,
    curr_pol: Array,
    curr_tor: Array,
    iota: Array,
    grid: Dict[str, float],
    calc_cur: bool = False,
) -> SurfaceData:
    """Initialize a single flux surface: Fourier sums, derived fields, splines, B min/max."""
    fourier = fourier_sums(
        theta_arr,
        phi_arr,
        coeffs["rmnc"],
        coeffs["zmns"],
        coeffs["lmns"],
        coeffs["bmnc"],
        ixm,
        ixn,
        nfp=nfp,
        max_m_mode=max_m_mode,
        max_n_mode=max_n_mode,
        lasym=bool(coeffs.get("lasym", False)),
        rmns=coeffs.get("rmns"),
        zmnc=coeffs.get("zmnc"),
        lmnc=coeffs.get("lmnc"),
        bmns=coeffs.get("bmns"),
    )

    derived = derived_quantities(fourier, curr_pol=curr_pol, curr_tor=curr_tor, iota=iota)
    fields = {**fourier, **derived}

    splines = build_splines(fields, grid["theta_int"], grid["phi_int"], grid["mt"], grid["mp"], calc_cur)

    # Find initial min/max from grid, then refine with Newton.
    # Fortran's MINLOC/MAXLOC traverse arrays in column-major order
    # (theta index varies fastest). Use exact extrema and pick the
    # first index in Fortran order when there are ties.
    b = fields["b"]
    b_max = jnp.max(b)
    b_min = jnp.min(b)

    max_i, max_j = _select_extremum_index(
        b,
        theta_arr,
        phi_arr,
        ixm,
        ixn,
        coeffs["bmnc"],
        max_m_mode,
        max_n_mode,
        find_max=True,
    )
    min_i, min_j = _select_extremum_index(
        b,
        theta_arr,
        phi_arr,
        ixm,
        ixn,
        coeffs["bmnc"],
        max_m_mode,
        max_n_mode,
        find_max=False,
    )

    theta_bmin = theta_arr[min_i]
    phi_bmin = phi_arr[min_j]
    theta_bmax = theta_arr[max_i]
    phi_bmax = phi_arr[max_j]

    theta_bmin, phi_bmin, _it_min, _err_min = neo_zeros2d(
        theta_bmin, phi_bmin, 1.0e-10, 100, splines["b_spl"], grid
    )
    theta_bmax, phi_bmax, _it_max, _err_max = neo_zeros2d(
        theta_bmax, phi_bmax, 1.0e-10, 100, splines["b_spl"], grid
    )

    # Evaluate B at refined points.
    from .geometry import neo_eval

    b_min, *_ = neo_eval(
        theta_bmin, phi_bmin,
        splines["b_spl"],
        splines["g_spl"],
        splines["k_spl"],
        splines["p_spl"],
        splines.get("q_spl"),
        grid,
    )
    b_max, *_ = neo_eval(
        theta_bmax, phi_bmax,
        splines["b_spl"],
        splines["g_spl"],
        splines["k_spl"],
        splines["p_spl"],
        splines.get("q_spl"),
        grid,
    )

    bmref = b_max

    return SurfaceData(
        b_min=b_min,
        b_max=b_max,
        theta_bmin=theta_bmin,
        phi_bmin=phi_bmin,
        theta_bmax=theta_bmax,
        phi_bmax=phi_bmax,
        bmref=bmref,
        fields=fields,
        splines=splines,
    )
