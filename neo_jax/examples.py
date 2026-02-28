"""Example workflows for NEO_JAX."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import jax
import jax.numpy as jnp

from .api import load_boozmn, run_neo
from .config import NeoConfig
from .integrate import flint_bo_jax
from .plotting import plot_epsilon_effective
from .workflow import build_surface_problem


def _default_ncsx_boozmn_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"


def ncsx_jit_demo(
    *,
    boozmn_path: str | Path | None = None,
    surfaces: Sequence[int] | None = None,
    theta_n: int = 64,
    phi_n: int = 64,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Run a JIT-compiled NCSX demo and plot epsilon effective."""
    if boozmn_path is None:
        boozmn_path = _default_ncsx_boozmn_path()

    if surfaces is None:
        surfaces = [19, 39, 59, 79]

    config = NeoConfig(
        surfaces=list(surfaces),
        theta_n=theta_n,
        phi_n=phi_n,
    )

    results = run_neo(boozmn_path, config=config, use_jax=True, progress=True)

    fig, _ax = plot_epsilon_effective(results, label="NEO_JAX")
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        import matplotlib.pyplot as plt

        plt.show()


def ncsx_autodiff_demo(
    *,
    boozmn_path: str | Path | None = None,
    surface: int | float = 0.35,
    steps: int = 5,
    lr: float = 1.0e-2,
) -> None:
    """Autodiff demo: optimize Rmajor to hit a target epsilon effective."""
    if boozmn_path is None:
        boozmn_path = _default_ncsx_boozmn_path()

    booz = load_boozmn(boozmn_path)
    config = NeoConfig(surfaces=[surface], theta_n=64, phi_n=64)
    problem = build_surface_problem(booz, config, surface=surface)
    Rmajor_base = jnp.asarray(problem.Rmajor)

    def epstot_from_Rmajor(Rmajor: jnp.ndarray) -> jnp.ndarray:
        out = flint_bo_jax(problem.surface, problem.params, problem.env, nfp=booz.nfp, Rmajor=Rmajor)
        return out["epstot"]

    epstot_fn = jax.jit(epstot_from_Rmajor)

    base_epstot = epstot_fn(Rmajor_base)
    target = 0.9 * base_epstot

    def loss(Rmajor: jnp.ndarray) -> jnp.ndarray:
        epstot = epstot_fn(Rmajor)
        return (epstot - target) ** 2

    grad_loss = jax.grad(loss)

    Rmajor = Rmajor_base
    for step in range(steps):
        loss_val = loss(Rmajor)
        grad_val = grad_loss(Rmajor)
        Rmajor = Rmajor - lr * grad_val
        print(f"step={step} Rmajor={float(Rmajor):.6e} loss={float(loss_val):.6e}")

    print("base epstot:", float(base_epstot))
    print("target epstot:", float(target))
    print("final epstot:", float(epstot_fn(Rmajor)))
