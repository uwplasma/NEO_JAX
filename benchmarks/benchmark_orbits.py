"""Benchmark the ORBITS reference case."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from neo_jax.control import read_control
from neo_jax.driver import run_neo_from_boozmn


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NEO_JAX on ORBITS")
    parser.add_argument(
        "--control",
        default="tests/fixtures/orbits/neo_in.ORBITS",
        help="Path to NEO control file",
    )
    parser.add_argument(
        "--boozmn",
        default="tests/fixtures/orbits/boozmn_ORBITS.nc",
        help="Path to boozmn file",
    )
    parser.add_argument("--jax", action="store_true", help="Use JAX scan backend")
    parser.add_argument("--warmup", action="store_true", help="Run a warmup iteration before timing")

    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    control_path = Path(args.control)
    boozmn_path = Path(args.boozmn)
    if not control_path.is_absolute():
        control_path = (repo_root / control_path).resolve()
    if not boozmn_path.is_absolute():
        boozmn_path = (repo_root / boozmn_path).resolve()

    control = read_control(control_path)
    if args.warmup:
        import jax
        import jax.numpy as jnp

        warmup = run_neo_from_boozmn(str(boozmn_path), control, use_jax=args.jax)
        if warmup:
            diag = warmup[-1].get("diagnostics", {})
            arr = diag.get("bigint")
            if arr is None:
                arr = jnp.asarray(warmup[-1]["epstot"])
            jax.block_until_ready(arr)

    t0 = time.perf_counter()
    results = run_neo_from_boozmn(str(boozmn_path), control, use_jax=args.jax)
    t1 = time.perf_counter()

    dt = t1 - t0
    per_surface = dt / max(1, len(results))

    print(f"Surfaces: {len(results)}")
    print(f"Total time: {dt:.3f} s")
    print(f"Per surface: {per_surface:.3f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
