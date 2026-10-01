"""Benchmark the ORBITS reference case."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, replace
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

    parser.add_argument("--no-diagnostics", action="store_true", help="Disable progress and history output")
    parser.add_argument("--reference-bin", type=Path, help="STELLOPT xneo executable")
    parser.add_argument("--output", type=Path, help="Append timings and ripple values to JSON")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    control_path = Path(args.control)
    boozmn_path = Path(args.boozmn)
    if not control_path.is_absolute():
        control_path = (repo_root / control_path).resolve()
    if not boozmn_path.is_absolute():
        boozmn_path = (repo_root / boozmn_path).resolve()

    control = read_control(control_path)
    if args.no_diagnostics:
        control = replace(control, write_progress=0, write_output_files=0,
                          write_integrate=0, write_diagnostic=0)
    reference_s = None
    reference = None
    if args.reference_bin:
        import numpy as np

        if control.inp_swi != 0 or control.in_file != "boozmn" or not control.fluxs_arr:
            parser.error("Reference comparisons require inp_swi=0, IN_FILE=boozmn and explicit surfaces")
        extension = control_path.name.split(".", 1)[-1]
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            values = list(asdict(control).values())
            body = [*map(str, values[:2]), str(len(values[2])),
                    " ".join(map(str, values[2])),
                    *(format(v, ".15g") for v in values[3:24]),
                    "0", "0", "0", *(format(v, ".15g") if isinstance(v, float)
                                      else str(v) for v in values[24:])]
            (work / f"neo_in.{extension}").write_text("! Benchmark\n!\n!\n" + "\n".join(body) + "\n")
            shutil.copy2(boozmn_path, work / f"boozmn_{extension}.nc")
            start = time.perf_counter()
            binary = shutil.which(str(args.reference_bin)) or str(args.reference_bin.resolve())
            subprocess.run([binary, extension], cwd=work,
                           check=True, stdout=subprocess.DEVNULL)
            reference_s = time.perf_counter() - start
            reference = np.atleast_2d(np.loadtxt(work / control.out_file))[:, 1]

    import jax

    timings = []
    for _ in range(2 if args.warmup else 1):
        start = time.perf_counter()
        results = run_neo_from_boozmn(str(boozmn_path), control, use_jax=args.jax)
        jax.block_until_ready(results.epsilon_effective)
        timings.append(time.perf_counter() - start)
    dt = timings[-1]
    if args.output:
        records = json.loads(args.output.read_text()) if args.output.exists() else []
        records.append({
            "case": control_path.name.removeprefix("neo_in."), "surfaces": len(results),
            "controls": asdict(control),
            "jax_version": jax.__version__, "backend": jax.default_backend(),
            "jax_first_s": timings[0], "jax_warm_s": timings[-1] if args.warmup else None,
            "fortran_cli_s": reference_s, "s": results.s.tolist(),
            "jax_epstot": results.epsilon_effective.tolist(),
            "fortran_epstot": reference.tolist() if reference is not None else None,
        })
        args.output.write_text(json.dumps(records, indent=2) + "\n")
    per_surface = dt / max(1, len(results))

    print(f"Surfaces: {len(results)}")
    print(f"Total time: {dt:.3f} s")
    print(f"Per surface: {per_surface:.3f} s")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
