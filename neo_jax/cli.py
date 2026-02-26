"""Command-line interface for NEO_JAX."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .control import read_control
from .driver import run_neo_from_boozmn
from .io import resolve_boozmn_path, resolve_control_path


def _format_line(result: dict, eout_swi: int) -> str:
    flux_index = int(result["flux_index"])
    if eout_swi == 1:
        return (
            f"{flux_index:8d}"
            f" {result['epstot']: .10e}"
            f" {result['reff']: .10e}"
            f" {result['iota']: .10e}"
            f" {result['b_ref']: .10e}"
            f" {result['r_ref']: .10e}"
        )
    if eout_swi == 2:
        epspar = result["epspar"]
        epspar_1 = float(epspar[0]) if len(epspar) > 0 else 0.0
        epspar_2 = float(epspar[1]) if len(epspar) > 1 else 0.0
        return (
            f"{flux_index:8d}"
            f" {result['epstot']: .10e}"
            f" {result['reff']: .10e}"
            f" {result['iota']: .10e}"
            f" {result['b_ref']: .10e}"
            f" {result['r_ref']: .10e}"
            f" {epspar_1: .10e}"
            f" {epspar_2: .10e}"
            f" {result['ctrone']: .10e}"
            f" {result['ctrtot']: .10e}"
            f" {result['bareph']: .10e}"
            f" {result['barept']: .10e}"
            f" {result['yps']: .10e}"
        )
    if eout_swi == 10:
        return f"{result['b_ref']: .10e} {result['r_ref']: .10e} {result['epstot']: .10e}"
    raise ValueError(f"Unsupported eout_swi: {eout_swi}")


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEO_JAX CLI (xneo-compatible).")
    parser.add_argument("extension", nargs="?", help="NEO input extension (xneo style)")
    parser.add_argument("--control", help="Path to NEO control file")
    parser.add_argument("--boozmn", help="Path to boozmn file (netCDF)")
    parser.add_argument("--jax", action="store_true", help="Use JAX scan backend")
    parser.add_argument("--output", help="Override NEO output file path")
    parser.add_argument("--verbose", action="store_true", help="Print progress to stdout")

    args = parser.parse_args(argv)

    if args.control:
        control_path = Path(args.control)
    else:
        control_path = resolve_control_path(args.extension)

    control = read_control(control_path)

    if args.boozmn:
        booz_base = args.boozmn
    else:
        booz_base = "boozmn" if control.inp_swi == 0 else control.in_file
    boozmn_path = resolve_boozmn_path(booz_base, args.extension)

    if args.verbose:
        print(f"NEO_JAX: control={control_path} boozmn={boozmn_path}")

    results = run_neo_from_boozmn(str(boozmn_path), control, use_jax=args.jax, progress=args.verbose)

    lines = [_format_line(result, control.eout_swi) for result in results]

    output_path = Path(args.output) if args.output else Path(control.out_file)
    _write_lines(output_path, lines)

    if args.verbose:
        print(f"NEO_JAX: wrote {output_path}")
        for line in lines:
            print(line)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
