"""Plotting helpers for NEO_JAX outputs."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .results import NeoResults


def plot_epsilon_effective(
    results: NeoResults,
    *,
    ax=None,
    label: str | None = None,
) -> Tuple[object, object]:
    """Plot epsilon effective vs effective radius.

    Returns (fig, ax). matplotlib is imported lazily.
    """
    import matplotlib.pyplot as plt  # lazy import

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    r_eff = np.asarray(results.reff)
    eps_eff = np.asarray(results.epsilon_effective)

    ax.plot(r_eff, eps_eff, marker="o", label=label)
    ax.set_xlabel("r_eff")
    ax.set_ylabel("epsilon_effective")
    if label:
        ax.legend()
    return fig, ax
