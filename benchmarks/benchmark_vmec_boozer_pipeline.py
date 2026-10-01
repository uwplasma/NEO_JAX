"""Benchmark the JAX-native VMEC→Boozer→NEO pipeline with JIT reuse."""

from __future__ import annotations

import argparse
import time


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark JIT reuse for the VMEC→Boozer→NEO pipeline")
    parser.add_argument("input", help="VMEC input file")
    parser.add_argument("--surfaces", default="0.4,0.6,0.8", help="Comma-separated s values in [0,1]")
    parser.add_argument("--theta-n", type=int, default=16, help="NEO theta grid size")
    parser.add_argument("--phi-n", type=int, default=16, help="NEO phi grid size")
    parser.add_argument("--mboz", type=int, default=6, help="Boozer m resolution")
    parser.add_argument("--nboz", type=int, default=6, help="Boozer n resolution")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repeats after JIT warmup")
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

    print("Input:", args.input)
    print("Surfaces:", args.surfaces)
    print(f"JIT compile + first run: {compile_time:.3f} s")
    print(f"Mean reuse time ({len(timings)} runs): {mean_time:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
