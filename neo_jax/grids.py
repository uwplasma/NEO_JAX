"""Grid preparation utilities."""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def prepare_grids(theta_n: int, phi_n: int, nfp: int) -> Dict[str, jnp.ndarray | float | int]:
    """Prepare theta/phi grids and spacing.

    Mirrors `neo_prep.f90` grid construction.
    """
    theta_start = 0.0
    theta_end = 2.0 * jnp.pi
    phi_start = 0.0
    phi_end = 2.0 * jnp.pi / nfp

    theta_int = (theta_end - theta_start) / (theta_n - 1)
    phi_int = (phi_end - phi_start) / (phi_n - 1)

    theta_arr = theta_start + theta_int * jnp.arange(theta_n)
    phi_arr = phi_start + phi_int * jnp.arange(phi_n)

    def _maybe_float(value):
        if isinstance(value, jax.Array):
            return value
        return float(value)

    return {
        "theta_start": theta_start,
        "theta_end": theta_end,
        "phi_start": phi_start,
        "phi_end": phi_end,
        "theta_int": _maybe_float(theta_int),
        "phi_int": _maybe_float(phi_int),
        "theta_arr": theta_arr,
        "phi_arr": phi_arr,
        "theta_n": theta_n,
        "phi_n": phi_n,
        "mt": 1,
        "mp": 1,
    }
