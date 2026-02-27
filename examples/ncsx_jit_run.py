"""JIT-compiled NCSX demo with plotting and API usage."""

from __future__ import annotations

from pathlib import Path

from neo_jax import NeoConfig, plot_epsilon_effective, run_boozmn


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    boozmn_path = repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"

    config = NeoConfig(
        surfaces=[19, 39, 59, 79],
        theta_n=64,
        phi_n=64,
        write_progress=True,
    )

    results = run_boozmn(boozmn_path, config=config, use_jax=True)

    # Access by alias (arrays over surfaces)
    print("epsilon_effective:", results.epsilon_effective)
    print("reff:", results.reff)

    # Access a single surface by name
    first = results[0]
    print("surface", first.flux_index, "epsilon_effective:", first.epsilon_effective)
    print("epspar shape:", first.epsilon_effective_by_class.shape)

    # Plot epsilon effective vs radius
    fig, _ax = plot_epsilon_effective(results, label="NEO_JAX")
    out_path = Path("ncsx_eps_eff.png")
    fig.savefig(out_path, dpi=150)
    print("saved plot to", out_path.resolve())


if __name__ == "__main__":
    main()
