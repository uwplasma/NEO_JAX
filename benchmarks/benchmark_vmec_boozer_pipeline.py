"""Benchmark the JAX-native VMEC→Boozer→NEO pipeline with JIT reuse."""

from __future__ import annotations

import argparse
import time
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
    parser = argparse.ArgumentParser(description="Benchmark JIT reuse for the VMEC→Boozer→NEO pipeline")
    parser.add_argument("--case", default="circular_tokamak", help="vmec_jax example case name")
    parser.add_argument("--surfaces", default="0.4,0.6,0.8", help="Comma-separated s values in [0,1]")
    parser.add_argument("--theta-n", type=int, default=16, help="NEO theta grid size")
    parser.add_argument("--phi-n", type=int, default=16, help="NEO phi grid size")
    parser.add_argument("--mboz", type=int, default=6, help="Boozer m resolution")
    parser.add_argument("--nboz", type=int, default=6, help="Boozer n resolution")
    parser.add_argument("--ntheta", type=int, default=16, help="VMEC theta grid size")
    parser.add_argument("--nzeta", type=int, default=1, help="VMEC zeta grid size")
    parser.add_argument("--max-iter", type=int, default=2, help="VMEC iterations per solve")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats after JIT warmup")
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

    t0 = time.perf_counter()
    solver = build_vmec_boozer_neo_jax(
        run,
        booz_kwargs=booz_kwargs,
        neo_config=config,
        jit=True,
    )
    outputs = solver(run.state)
    jax.block_until_ready(jnp.asarray(outputs.eps_eff))
    t1 = time.perf_counter()
    compile_time = t1 - t0

    timings = []
    for _ in range(int(args.repeats)):
        t_start = time.perf_counter()
        outputs = solver(run.state)
        jax.block_until_ready(jnp.asarray(outputs.eps_eff))
        timings.append(time.perf_counter() - t_start)

    mean_time = sum(timings) / max(1, len(timings))

    print("Case:", args.case)
    print("Surfaces:", args.surfaces)
    print(f"JIT compile + first run: {compile_time:.3f} s")
    print(f"Mean reuse time ({len(timings)} runs): {mean_time:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
