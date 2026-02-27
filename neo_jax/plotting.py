"""Plotting helpers for NEO_JAX outputs."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .results import NeoResults


def plot_epsilon_effective(
    results: NeoResults,
    *,
    ax=None,
    x: str = "s",
    label: str | None = None,
) -> Tuple[object, object]:
    """Plot epsilon effective vs a radial coordinate.

    Parameters
    ----------
    x : {"s", "sqrt_s", "r_eff"}
        Radial coordinate on the x-axis. Default is ``s``.

    Returns (fig, ax). matplotlib is imported lazily.
    """
    import matplotlib.pyplot as plt  # lazy import

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    x_key = x.lower()
    if x_key in {"s", "psi"}:
        x_vals = np.asarray(results.s)
        xlabel = "s"
    elif x_key in {"sqrt_s", "sqrt(s)"}:
        x_vals = np.asarray(results.sqrt_s)
        xlabel = "sqrt(s)"
    elif x_key in {"r_eff", "reff"}:
        x_vals = np.asarray(results.r_eff)
        xlabel = "r_eff"
    else:
        raise ValueError(f"Unsupported x-axis '{x}'")

    eps_eff = np.asarray(results.epsilon_effective)

    ax.plot(x_vals, eps_eff, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("epsilon_effective")
    if label:
        ax.legend()
    return fig, ax
