"""Autodiff optimization demo: adjust Rmajor to reduce epsilon effective.

This is a toy example that optimizes a single scalar (Rmajor) using a
SciPy least-squares solver with JAX gradients. Real optimizations should
use vmec_jax boundary coefficients.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from neo_jax import NeoConfig, build_surface_problem, run_neo
from neo_jax.integrate import flint_bo_jax


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    boozmn_path = repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"

    # Choose the target surface by normalized toroidal flux s.
    # These are mapped to the nearest surface in the boozmn file.
    config = NeoConfig(
        surfaces=[0.35],
        theta_n=64,
        phi_n=64,
        # Integration controls (matching the NEO control file):
        npart=40,  # number of eta grid points
        multra=2,  # number of trapped particle classes to resolve
        nstep_per=20,  # RK4 steps per field period
        nstep_min=200,
        nstep_max=500,
        acc_req=0.02,  # rational-surface accuracy requirement
        no_bins=50,  # bins for rational-surface coverage
    )
    baseline = run_neo(boozmn_path, config=config, use_jax=True)
    print("baseline epsilon_effective:", baseline.epsilon_effective)

    # Build a surface problem so users don't have to wire init_surface/RhsEnv manually.
    # Load only the selected surface so the reference matches the baseline run.
    from neo_jax.api import load_boozmn

    surface_index = int(baseline[0].flux_index)
    booz = load_boozmn(boozmn_path, surfaces=[surface_index])
    problem = build_surface_problem(booz, config, surface=config.surfaces[0])

    # Use Rmajor as the optimization variable.
    Rmajor0 = problem.Rmajor

    def eps_eff_from_Rmajor(Rmajor: jnp.ndarray) -> jnp.ndarray:
        out = flint_bo_jax(problem.surface, problem.params, problem.env, nfp=booz.nfp, rt0=Rmajor)
        return out["epstot"] * problem.scale

    eps_fn = jax.jit(eps_eff_from_Rmajor)
    eps0 = float(eps_fn(jnp.asarray(Rmajor0)))
    print("baseline epsilon_effective (low-level):", eps0)
    target = 0.9 * eps0

    residual_scale = 1.0e4

    def residual(Rmajor_np: np.ndarray) -> np.ndarray:
        Rmajor = jnp.asarray(Rmajor_np[0])
        res = eps_fn(Rmajor) - target
        return np.array([float(res * residual_scale)])

    def jacobian(Rmajor_np: np.ndarray) -> np.ndarray:
        Rmajor = jnp.asarray(Rmajor_np[0])
        grad_val = jax.grad(lambda x: eps_fn(x))(Rmajor)
        return np.array([[float(grad_val * residual_scale)]])

    print("target epsilon_effective:", target)
    # Replace least_squares with your preferred optimizer (jaxopt, optax, etc.)
    # while reusing the same residual/jacobian callbacks.
    result = least_squares(
        residual,
        x0=np.array([Rmajor0]),
        jac=jacobian,
        gtol=1.0e-14,
        ftol=1.0e-14,
        xtol=1.0e-14,
        max_nfev=20,
    )
    print("optimizer status:", result.message)
    print("optimized Rmajor:", result.x[0])
    print("final epsilon_effective:", float(eps_fn(jnp.asarray(result.x[0]))))


if __name__ == "__main__":
    main()
