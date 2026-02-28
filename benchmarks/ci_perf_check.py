"""Small perf regression check for the JAX VMEC→Boozer→NEO pipeline."""

from __future__ import annotations

import os
import time


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def main() -> int:
    case = os.getenv("NEO_JAX_CI_PERF_CASE", "circular_tokamak")
    surfaces = os.getenv("NEO_JAX_CI_PERF_SURFACES", "0.5")
    compile_max = _env_float("NEO_JAX_CI_PERF_COMPILE_MAX", 60.0)
    reuse_max = _env_float("NEO_JAX_CI_PERF_REUSE_MAX", 0.5)
    repeats = _env_int("NEO_JAX_CI_PERF_REPEATS", 1)

    import jax
    import jax.numpy as jnp

    import vmec_jax as vj
    from vmec_jax.driver import example_paths
    from vmec_jax.vmec_tomnsp import vmec_angle_grid

    from neo_jax import NeoConfig, build_vmec_boozer_neo_jax

    input_path, _ = example_paths(case)
    cfg, _ = vj.load_input(str(input_path))
    grid = vmec_angle_grid(
        ntheta=8,
        nzeta=1,
        nfp=int(cfg.nfp),
        lasym=bool(cfg.lasym),
    )

    vmec_kwargs = dict(
        max_iter=1,
        use_initial_guess=True,
        vmec_project=False,
        verbose=False,
        grid=grid,
    )

    run = vj.run_fixed_boundary(input_path, **vmec_kwargs)
    config = NeoConfig(
        theta_n=8,
        phi_n=8,
        surfaces=[float(s) for s in surfaces.split(",") if s.strip()],
        npart=8,
        multra=1,
        nstep_per=4,
        nstep_min=20,
        nstep_max=40,
        acc_req=0.2,
        no_bins=10,
    )

    t0 = time.perf_counter()
    solver = build_vmec_boozer_neo_jax(
        run,
        booz_kwargs=dict(mboz=4, nboz=4),
        neo_config=config,
        jit=True,
    )
    outputs = solver(run.state)
    jax.block_until_ready(jnp.asarray(outputs.eps_eff))
    compile_time = time.perf_counter() - t0

    timings = []
    for _ in range(max(1, repeats)):
        t_start = time.perf_counter()
        outputs = solver(run.state)
        jax.block_until_ready(jnp.asarray(outputs.eps_eff))
        timings.append(time.perf_counter() - t_start)

    reuse_time = sum(timings) / len(timings)

    print("CI perf check")
    print("Case:", case)
    print("Surfaces:", surfaces)
    print(f"Compile+first: {compile_time:.3f} s (max {compile_max:.3f} s)")
    print(f"Mean reuse: {reuse_time:.3f} s (max {reuse_max:.3f} s)")

    if compile_time > compile_max:
        print("FAIL: compile time exceeded threshold.")
        return 1
    if reuse_time > reuse_max:
        print("FAIL: reuse time exceeded threshold.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
