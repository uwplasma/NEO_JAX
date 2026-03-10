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


def _describe_jax_runtime() -> str:
    """Return a concise runtime summary for progress logging."""
    try:
        import jax

        devices = jax.devices()
        backend = jax.default_backend()
        if not devices:
            return f"{backend} (no devices reported)"
        sample = ", ".join(getattr(dev, "device_kind", dev.platform) for dev in devices[:2])
        if len(devices) > 2:
            sample += ", ..."
        return f"{backend} ({len(devices)} device{'s' if len(devices) != 1 else ''}: {sample})"
    except Exception as exc:  # pragma: no cover - defensive logging only
        return f"unavailable ({exc})"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NEO_JAX CLI with legacy xneo-compatible defaults.")
    parser.add_argument("extension", nargs="?", help="Legacy NEO input extension (same as xneo)")
    parser.add_argument("--control", help="Path to a specific NEO control file")
    parser.add_argument("--boozmn", help="Override the boozmn input path")
    parser.add_argument("--jax", action="store_true", help="Prefer the JAX backend when compatible")
    parser.add_argument("--no-jax", action="store_true", help="Force the Python backend")
    parser.add_argument("--output", help="Override the main NEO output filename")
    parser.add_argument("--verbose", action="store_true", help="Print additional progress information")
    parser.add_argument("--quiet", action="store_true", help="Suppress NEO_JAX progress messages")
    parser.set_defaults(jax=True)
    args = parser.parse_args(argv)

    control_path = Path(args.control) if args.control else resolve_control_path(args.extension)
    control = read_control(control_path)

    progress = not bool(args.quiet)
    boozmn_path = _resolve_boozmn_for_xneo(control, args.extension, args.boozmn)
    surface_count = len(control.fluxs_arr) if control.fluxs_arr else "all"
    backend = "JAX" if bool(args.jax) and not bool(args.no_jax) else "Python"
    output_path = Path(args.output) if args.output else Path(control.out_file)

    if progress:
        print("NEO_JAX: starting legacy CLI solve")
        print(f"NEO_JAX: control={control_path}")
        print(f"NEO_JAX: boozmn={boozmn_path}")
        print(
            "NEO_JAX: mode="
            f"{'calc_cur=1 (epsilon effective + parallel current)' if control.calc_cur else 'calc_cur=0 (epsilon effective)'}"
        )
        print(
            "NEO_JAX: surfaces="
            f"{surface_count} theta_n={control.theta_n} phi_n={control.phi_n} "
            f"npart={control.npart} backend={backend}"
        )
        if backend == "JAX":
            print(f"NEO_JAX: jax_runtime={_describe_jax_runtime()}")
        if control.write_output_files or control.write_integrate or control.write_diagnostic or control.write_cur_inte:
            print(
                "NEO_JAX: legacy extras="
                f"write_output_files={control.write_output_files} "
                f"write_integrate={control.write_integrate} "
                f"write_diagnostic={control.write_diagnostic} "
                f"write_cur_inte={control.write_cur_inte}"
            )
        if control.write_integrate:
            print("NEO_JAX: WRITE_INTEGRATE=1 enabled; writing conver.dat from the parity solve.")
        if control.calc_cur:
            print(f"NEO_JAX: current output will be written to {control.cur_file}")

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
    _write_lines(output_path, lines)

    if progress:
        print(f"NEO_JAX: wrote {output_path}")
        if control.calc_cur:
            print(f"NEO_JAX: wrote {control.cur_file}")
        if control.write_cur_inte:
            print("NEO_JAX: wrote current.dat")
        if control.write_integrate:
            print("NEO_JAX: wrote conver.dat")
        if control.write_diagnostic:
            print("NEO_JAX: wrote diagnostic.dat, diagnostic_add.dat, diagnostic_bigint.dat")
        for line in lines:
            print(line)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
