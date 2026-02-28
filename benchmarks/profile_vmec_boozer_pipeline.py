"""Profile the JAX-native VMEC→Boozer→NEO pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_surfaces(text: str) -> list[float]:
    vals = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        vals.append(float(item))
    return vals


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile VMEC→Boozer→NEO pipeline")
    parser.add_argument("--case", default="circular_tokamak", help="vmec_jax example case name")
    parser.add_argument("--surfaces", default="0.4,0.6,0.8", help="Comma-separated s values in [0,1]")
    parser.add_argument("--theta-n", type=int, default=16, help="NEO theta grid size")
    parser.add_argument("--phi-n", type=int, default=16, help="NEO phi grid size")
    parser.add_argument("--mboz", type=int, default=6, help="Boozer m resolution")
    parser.add_argument("--nboz", type=int, default=6, help="Boozer n resolution")
    parser.add_argument("--ntheta", type=int, default=16, help="VMEC theta grid size")
    parser.add_argument("--nzeta", type=int, default=1, help="VMEC zeta grid size")
    parser.add_argument("--max-iter", type=int, default=2, help="VMEC iterations per solve")
    parser.add_argument("--trace-dir", default="profiles/vmec_boozer_neo_trace", help="JAX trace output dir")
    parser.add_argument("--hlo-out", default="profiles/vmec_boozer_neo.hlo.txt", help="Path to write HLO")
    parser.add_argument("--skip-hlo", action="store_true", help="Skip HLO dump")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    import vmec_jax as vj
    from vmec_jax.driver import example_paths
    from vmec_jax.vmec_tomnsp import vmec_angle_grid

    from neo_jax import NeoConfig, build_vmec_boozer_neo_jax

    input_path, _ = example_paths(args.case)
    cfg, _ = vj.load_input(str(input_path))
    grid = vmec_angle_grid(
        ntheta=int(args.ntheta),
        nzeta=int(args.nzeta),
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )

    vmec_kwargs = dict(
        max_iter=int(args.max_iter),
        use_initial_guess=True,
        vmec_project=False,
        verbose=False,
        grid=grid,
    )

    run = vj.run_fixed_boundary(input_path, **vmec_kwargs)
    config = NeoConfig(
        theta_n=int(args.theta_n),
        phi_n=int(args.phi_n),
        surfaces=_parse_surfaces(args.surfaces),
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
