"""Toy optimization: scale |B| spectrum to reduce epsilon effective.

This example demonstrates end-to-end autodiff through NEO_JAX by optimizing
a simple scalar (scale factor on bmnc). For real shape optimization, use
vmec_jax + booz_xform_jax once the JAX-native pipeline is wired.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from neo_jax import NeoConfig, run_neo
from neo_jax.api import load_boozmn


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    boozmn_path = repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"

    # Use an explicit surface index to avoid s-mapping inside JAX traces.
    surface_index = 36
    config = NeoConfig(surfaces=[surface_index], theta_n=64, phi_n=64)

    booz = load_boozmn(boozmn_path, surfaces=[surface_index])

    def eps_eff_from_scale(scale: jnp.ndarray) -> jnp.ndarray:
        scaled = replace(booz, bmnc=booz.bmnc * scale)
        outputs = run_neo(scaled, config=config, use_jax=True, jax_surface_scan=True)
        return outputs.eps_eff[0]

    eps_fn = jax.jit(eps_eff_from_scale)
    eps0 = float(eps_fn(jnp.array(1.0)))
    print("baseline epsilon_effective:", eps0)

    target = 0.9 * eps0
    print("target epsilon_effective:", target)

    scale = jnp.array(1.0)
    lr = 0.1
    for step in range(5):
        loss = (eps_fn(scale) - target) ** 2
        grad = jax.grad(lambda s: (eps_fn(s) - target) ** 2)(scale)
        scale = scale - lr * grad
        print(f"step={step} scale={float(scale):.6f} loss={float(loss):.6e}")

    print("optimized scale:", float(scale))
    print("final epsilon_effective:", float(eps_fn(scale)))


if __name__ == "__main__":
    main()
