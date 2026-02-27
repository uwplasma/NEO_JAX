"""Compute epsilon effective on NCSX and plot versus s (default)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from neo_jax import NeoConfig, load_boozmn, plot_epsilon_effective, run_neo


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    boozmn_path = repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"

    # Surfaces can be specified by normalized toroidal flux s in [0, 1].
    # These are mapped to the nearest available surface in the boozmn file.
    config = NeoConfig(
        surfaces=[0.15, 0.35, 0.6, 0.85],
        theta_n=64,
        phi_n=64,
        write_progress=True,
    )

    results = run_neo(boozmn_path, config=config, use_jax=True)

    # Alternative: load the boozmn file once and reuse the BoozerData object.
    # Here we load the same surfaces selected above so the comparison is apples-to-apples.
    booz = load_boozmn(boozmn_path, surfaces=results.flux_index)
    results_from_booz = run_neo(booz, config=NeoConfig(theta_n=64, phi_n=64), use_jax=True)
    assert np.allclose(results_from_booz.epsilon_effective, results.epsilon_effective)

    # Access by name (aliases supported)
    print("s:", results.s)
    print("epsilon_effective:", results.epsilon_effective)

    # Access a single surface by name
    first = results[0]
    print("surface", first.flux_index, "epsilon_effective:", first.epsilon_effective)
    print("epspar shape:", first.epsilon_effective_by_class.shape)

    # Plot epsilon effective vs s (default), or set x="r_eff" / x="sqrt_s".
    fig, _ax = plot_epsilon_effective(results, label="NEO_JAX", x="s")
    out_path = Path(__file__).with_name("ncsx_eps_eff_vs_s.png")
    fig.savefig(out_path, dpi=150)
    print("saved plot to", out_path.resolve())


if __name__ == "__main__":
    main()
