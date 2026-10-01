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
    input_path = os.getenv("NEO_JAX_CI_PERF_INPUT")
    surfaces = os.getenv("NEO_JAX_CI_PERF_SURFACES", "0.5")
    compile_max = _env_float("NEO_JAX_CI_PERF_COMPILE_MAX", 60.0)
    reuse_max = _env_float("NEO_JAX_CI_PERF_REUSE_MAX", 0.5)
    repeats = _env_int("NEO_JAX_CI_PERF_REPEATS", 1)

    import jax
    import jax.numpy as jnp

    from vmex import VmecInput
    from vmex import optimize
    from neo_jax import NeoConfig, build_vmec_boozer_neo_jax

    inp = VmecInput.from_file(input_path) if input_path else VmecInput(
        mpol=4, ntor=0, ns_array=[9], ntheta=16,
        rbc=[[6, 1, 0, 0]], zbs=[[0, 1, 0, 0]], ai=[0.43], phiedge=6,
        niter_array=[2000], ftol_array=[1e-10], delt=0.9,
    )
    run = optimize.solve_equilibrium(inp)
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
        booz_kwargs=dict(mboz=4, nboz=0),
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
    print("Input:", input_path or "circular tokamak")
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
