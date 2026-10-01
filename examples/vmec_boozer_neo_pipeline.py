#!/usr/bin/env python3
"""Evaluate effective ripple from a symmetric VMEX input without intermediate files."""

from __future__ import annotations

import argparse

from vmex import VmecInput
from vmex import optimize
from neo_jax import NeoConfig, build_vmec_boozer_neo_jax, neo_outputs_to_results, plot_epsilon_effective


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="VMEC input file")
    parser.add_argument("--surfaces", default="0.4,0.6,0.8", help="Comma-separated normalized fluxes")
    parser.add_argument("--mboz", type=int, default=8)
    parser.add_argument("--nboz", type=int, default=8)
    parser.add_argument("--theta-n", type=int, default=32)
    parser.add_argument("--phi-n", type=int, default=32)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()

    eq = optimize.solve_equilibrium(VmecInput.from_file(args.input))
    cfg = NeoConfig(surfaces=[float(s) for s in args.surfaces.split(",")],
                    theta_n=args.theta_n, phi_n=args.phi_n)
    solver = build_vmec_boozer_neo_jax(
        eq, booz_kwargs=dict(mboz=args.mboz, nboz=args.nboz), neo_config=cfg)
    results = neo_outputs_to_results(solver(eq.state))
    plot_epsilon_effective(results, x="s")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
