"""Benchmark the NCSX reference case."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from neo_jax.control import read_control
from neo_jax.driver import run_neo_from_boozmn


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark NEO_JAX on NCSX")
    parser.add_argument(
        "--control",
        default="tests/fixtures/ncsx/neo_in.ncsx_c09r00_free",
        help="Path to NEO control file",
    )
    parser.add_argument(
        "--boozmn",
        default="tests/fixtures/ncsx/boozmn_ncsx_c09r00_free.nc",
        help="Path to boozmn file",
    )
    parser.add_argument("--jax", action="store_true", help="Use JAX scan backend")

    args = parser.parse_args()

    control = read_control(Path(args.control))
    t0 = time.perf_counter()
    results = run_neo_from_boozmn(args.boozmn, control, use_jax=args.jax)
    t1 = time.perf_counter()

    dt = t1 - t0
    per_surface = dt / max(1, len(results))

    print(f"Surfaces: {len(results)}")
    print(f"Total time: {dt:.3f} s")
    print(f"Per surface: {per_surface:.3f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
