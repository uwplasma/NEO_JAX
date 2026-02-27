"""Compute epsilon effective on NCSX and plot versus s (default)."""

from __future__ import annotations

from pathlib import Path

from neo_jax import NeoConfig, plot_epsilon_effective, run_boozmn


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

    results = run_boozmn(boozmn_path, config=config, use_jax=True)

    # Access by name (aliases supported)
    print("s:", results.s)
    print("epsilon_effective:", results.epsilon_effective)

    # Access a single surface by name
    first = results[0]
    print("surface", first.flux_index, "epsilon_effective:", first.epsilon_effective)
    print("epspar shape:", first.epsilon_effective_by_class.shape)

    # Plot epsilon effective vs s (default), or set x="r_eff" / x="sqrt_s".
    fig, _ax = plot_epsilon_effective(results, label="NEO_JAX", x="s")
    out_path = Path("ncsx_eps_eff_vs_s.png")
    fig.savefig(out_path, dpi=150)
    print("saved plot to", out_path.resolve())


if __name__ == "__main__":
    main()
