"""Profile NEO_JAX with JAX traces and optional XLA dumps."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path


def _apply_xla_dump(dump_dir: Path) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    flags = os.environ.get("XLA_FLAGS", "")
    dump_flags = (
        f"--xla_dump_to={dump_dir} "
        "--xla_dump_hlo_as_text "
        "--xla_dump_hlo_as_html "
        "--xla_dump_hlo_as_proto"
    )
    os.environ["XLA_FLAGS"] = f"{flags} {dump_flags}".strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Profile NEO_JAX runs")
    parser.add_argument(
        "--control",
        default="tests/fixtures/orbits/neo_in.ORBITS_FAST",
        help="Path to NEO control file",
    )
    parser.add_argument(
        "--boozmn",
        default="tests/fixtures/orbits/boozmn_ORBITS_FAST.nc",
        help="Path to boozmn file",
    )
    parser.add_argument("--jax", action="store_true", help="Use JAX scan backend")
    parser.add_argument(
        "--trace-dir",
        help="Directory for JAX profiler traces (TensorBoard compatible)",
    )
    parser.add_argument(
        "--xla-dump-dir",
        help="Directory to dump XLA HLO/LLVM artifacts (set before JAX import)",
    )
    parser.add_argument("--warmup", action="store_true", help="Run a warmup iteration before profiling")
    parser.add_argument("--progress", action="store_true", help="Print per-surface progress")
    parser.add_argument(
        "--enable-x64",
        action="store_true",
        help="Enable JAX 64-bit mode for parity runs",
    )

    args = parser.parse_args()

    if args.xla_dump_dir:
        _apply_xla_dump(Path(args.xla_dump_dir).expanduser().resolve())

    if args.enable_x64:
        os.environ.setdefault("JAX_ENABLE_X64", "1")

    if args.trace_dir:
        trace_dir = Path(args.trace_dir).expanduser().resolve()
        trace_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp

    if args.enable_x64:
        jax.config.update("jax_enable_x64", True)

    from neo_jax.control import read_control
    from neo_jax.driver import run_neo_from_boozmn

    repo_root = Path(__file__).resolve().parents[1]
    control_path = Path(args.control)
    boozmn_path = Path(args.boozmn)
    if not control_path.is_absolute():
        control_path = (repo_root / control_path).resolve()
    if not boozmn_path.is_absolute():
        boozmn_path = (repo_root / boozmn_path).resolve()

    control = read_control(control_path)

    if args.warmup:
        warmup = run_neo_from_boozmn(str(boozmn_path), control, use_jax=args.jax, progress=args.progress)
        if warmup:
            diag = warmup[-1].get("diagnostics", {})
            arr = diag.get("bigint")
            if arr is None:
                arr = jnp.asarray(warmup[-1]["epstot"])
            jax.block_until_ready(arr)

    if args.trace_dir:
        jax.profiler.start_trace(str(trace_dir))

    t0 = time.perf_counter()
    results = run_neo_from_boozmn(str(boozmn_path), control, use_jax=args.jax, progress=args.progress)
    if results:
        diag = results[-1].get("diagnostics", {})
        arr = diag.get("bigint")
        if arr is None:
            arr = jnp.asarray(results[-1]["epstot"])
        jax.block_until_ready(arr)
    t1 = time.perf_counter()

    if args.trace_dir:
        jax.profiler.stop_trace()

    dt = t1 - t0
    per_surface = dt / max(1, len(results))
    print(f"Surfaces: {len(results)}")
    print(f"Total time: {dt:.3f} s")
    print(f"Per surface: {per_surface:.3f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
