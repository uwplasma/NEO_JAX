"""Command-line interface for NEO_JAX with legacy ``xneo`` compatibility."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

from .control import read_control
from .driver import run_neo_from_boozmn
from .io import resolve_boozmn_path, resolve_control_path
from .legacy import build_fortran_line


def _format_line(result: dict, eout_swi: int) -> str:
    flux_index = int(result["flux_index"])
    if eout_swi == 1:
        return build_fortran_line(
            (flux_index,),
            int_width=8,
            reals=(
                float(result["epstot"]),
                float(result["reff"]),
                float(result["iota"]),
                float(result["b_ref"]),
                float(result["r_ref"]),
            ),
            real_width=17,
            real_digits=10,
            real_letter="E",
        )
    if eout_swi == 2:
        epspar = result["epspar"]
        epspar_1 = float(epspar[0]) if len(epspar) > 0 else 0.0
        epspar_2 = float(epspar[1]) if len(epspar) > 1 else 0.0
        return build_fortran_line(
            (flux_index,),
            int_width=8,
            reals=(
                float(result["epstot"]),
                float(result["reff"]),
                float(result["iota"]),
                float(result["b_ref"]),
                float(result["r_ref"]),
                epspar_1,
                epspar_2,
                float(result["ctrone"]),
                float(result["ctrtot"]),
                float(result["bareph"]),
                float(result["barept"]),
                float(result["yps"]),
            ),
            real_width=17,
            real_digits=10,
            real_letter="E",
        )
    if eout_swi == 10:
        return " ".join(
            part.strip()
            for part in (
                build_fortran_line((), reals=(float(result["b_ref"]),), real_width=17, real_digits=10).strip(),
                build_fortran_line((), reals=(float(result["r_ref"]),), real_width=17, real_digits=10).strip(),
                build_fortran_line((), reals=(float(result["epstot"]),), real_width=17, real_digits=10).strip(),
            )
        )
    raise ValueError(f"Unsupported eout_swi: {eout_swi}")


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_boozmn_for_xneo(control, extension: str | None, override: str | None) -> Path:
    if override:
        return resolve_boozmn_path(override, extension)
    if control.inp_swi == 0:
        if extension is None:
            return resolve_boozmn_path(control.in_file or "boozmn", None)
        return resolve_boozmn_path("boozmn", extension)
    return resolve_boozmn_path(control.in_file, extension)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEO_JAX CLI with legacy xneo-compatible defaults.")
    parser.add_argument("extension", nargs="?", help="Legacy NEO input extension (same as xneo)")
    parser.add_argument("--control", help="Path to a specific NEO control file")
    parser.add_argument("--boozmn", help="Override the boozmn input path")
    parser.add_argument("--jax", action="store_true", help="Prefer the JAX backend when compatible")
    parser.add_argument("--no-jax", action="store_true", help="Force the Python backend")
    parser.add_argument("--output", help="Override the main NEO output filename")
    parser.add_argument("--verbose", action="store_true", help="Print additional progress information")
    parser.set_defaults(jax=True)
    args = parser.parse_args(argv)

    control_path = Path(args.control) if args.control else resolve_control_path(args.extension)
    control = read_control(control_path)

    progress = bool(control.write_progress) or bool(args.verbose)
    boozmn_path = _resolve_boozmn_for_xneo(control, args.extension, args.boozmn)

    if progress:
        print(f"NEO_JAX: control={control_path} boozmn={boozmn_path}")

    use_jax = bool(args.jax) and not bool(args.no_jax)
    results = run_neo_from_boozmn(
        str(boozmn_path),
        control,
        use_jax=use_jax,
        progress=progress,
        extension=args.extension,
        legacy_mode=True,
    )

    lines = [_format_line(result, control.eout_swi) for result in results]
    output_path = Path(args.output) if args.output else Path(control.out_file)
    _write_lines(output_path, lines)

    if progress:
        print(f"NEO_JAX: wrote {output_path}")
        for line in lines:
            print(line)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
