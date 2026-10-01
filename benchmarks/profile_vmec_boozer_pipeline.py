"""Profile the JAX-native VMEC→Boozer→NEO pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile VMEC→Boozer→NEO pipeline")
    parser.add_argument("input", help="VMEC input file")
    parser.add_argument("--surfaces", default="0.4,0.6,0.8", help="Comma-separated s values in [0,1]")
    parser.add_argument("--theta-n", type=int, default=16, help="NEO theta grid size")
    parser.add_argument("--phi-n", type=int, default=16, help="NEO phi grid size")
    parser.add_argument("--mboz", type=int, default=6, help="Boozer m resolution")
    parser.add_argument("--nboz", type=int, default=6, help="Boozer n resolution")
    parser.add_argument("--trace-dir", default="profiles/vmec_boozer_neo_trace", help="JAX trace output dir")
    parser.add_argument("--hlo-out", default="profiles/vmec_boozer_neo.hlo.txt", help="Path to write HLO")
    parser.add_argument("--skip-hlo", action="store_true", help="Skip HLO dump")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    from vmex import VmecInput
    from vmex import optimize
    from neo_jax import NeoConfig, build_vmec_boozer_neo_jax

    run = optimize.solve_equilibrium(VmecInput.from_file(args.input))
    config = NeoConfig(
        theta_n=int(args.theta_n),
        phi_n=int(args.phi_n),
        surfaces=[float(s) for s in args.surfaces.split(",")],
        npart=12,
        multra=1,
        nstep_per=6,
        nstep_min=30,
        nstep_max=60,
        acc_req=0.1,
        no_bins=20,
    )

    booz_kwargs = dict(mboz=int(args.mboz), nboz=int(args.nboz))

    solver = build_vmec_boozer_neo_jax(
        run,
        booz_kwargs=booz_kwargs,
        neo_config=config,
        jit=True,
    )

    if not args.skip_hlo:
        hlo_path = Path(args.hlo_out)
        hlo_path.parent.mkdir(parents=True, exist_ok=True)
        lowered = solver.lower(run.state)
        hlo = lowered.compiler_ir(dialect="hlo")
        if hasattr(hlo, "as_hlo_text"):
            hlo_text = hlo.as_hlo_text()
        else:  # pragma: no cover - fallback
            hlo_text = str(hlo)
        hlo_path.write_text(hlo_text)

    trace_dir = Path(args.trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(trace_dir))
    outputs = solver(run.state)
    jax.block_until_ready(jnp.asarray(outputs.eps_eff))
    jax.profiler.stop_trace()

    print("Trace written to:", trace_dir)
    if not args.skip_hlo:
        print("HLO written to:", Path(args.hlo_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
