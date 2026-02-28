#!/usr/bin/env python3
"""End-to-end vmec_jax -> booz_xform_jax -> neo_jax pipeline example."""

from __future__ import annotations

import argparse

from neo_jax import NeoConfig, plot_epsilon_effective, run_vmec_boozer_neo


def _parse_surfaces(text: str) -> list[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run vmec_jax -> booz_xform_jax -> neo_jax without file I/O."
    )
    parser.add_argument(
        "--case",
        type=str,
        default="circular_tokamak",
        help="VMEC example case name (vmec_jax/examples/data/input.<case>).",
    )
    parser.add_argument(
        "--surfaces",
        type=str,
        default="0.4,0.6,0.8",
        help="Comma-separated s values in [0,1] (default: 0.4,0.6,0.8).",
    )
    parser.add_argument("--theta-n", type=int, default=32, help="NEO theta grid size.")
    parser.add_argument("--phi-n", type=int, default=32, help="NEO phi grid size.")
    parser.add_argument("--mboz", type=int, default=8, help="Boozer m resolution.")
    parser.add_argument("--nboz", type=int, default=8, help="Boozer n resolution.")
    parser.add_argument("--ntheta", type=int, default=16, help="VMEC theta grid size.")
    parser.add_argument("--nzeta", type=int, default=1, help="VMEC zeta grid size.")
    parser.add_argument(
        "--max-iter",
        type=int,
        default=3,
        help="VMEC max iterations (increase for a converged equilibrium).",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not display plots.")
    args = parser.parse_args()

    try:
        import vmec_jax as vj
        from vmec_jax.driver import example_paths
        from vmec_jax.vmec_tomnsp import vmec_angle_grid
    except ImportError as exc:
        raise ImportError(
            "vmec_jax is required for this example. Install vmec_jax or add it to PYTHONPATH."
        ) from exc

    input_path, _ = example_paths(args.case)
    cfg, _ = vj.load_input(str(input_path))
    grid = vmec_angle_grid(
        ntheta=int(args.ntheta),
        nzeta=int(args.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )

    config = NeoConfig(
        theta_n=int(args.theta_n),
        phi_n=int(args.phi_n),
        surfaces=_parse_surfaces(args.surfaces),
        write_progress=True,
    )

    vmec_kwargs = dict(
        max_iter=int(args.max_iter),
        use_initial_guess=True,
        vmec_project=False,
        verbose=True,
        grid=grid,
    )

    booz_kwargs = dict(mboz=int(args.mboz), nboz=int(args.nboz), jit=True)

    results = run_vmec_boozer_neo(
        input_path,
        vmec_kwargs=vmec_kwargs,
        booz_kwargs=booz_kwargs,
        neo_config=config,
        progress=True,
        fast_bcovar=True,
    )

    import numpy as np

    if not np.all(np.isfinite(results.epsilon_effective)):
        print(
            "[warning] Non-finite epsilon_effective values detected. "
            "Increase --max-iter or choose different surfaces for a converged equilibrium."
        )

    fig, _ = plot_epsilon_effective(results, x="s")
    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
