"""Legacy STELLOPT-style CLI formatting and file writers for NEO_JAX."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


def _normalize_mantissa(value: float, digits: int) -> tuple[float, int]:
    """Return a Fortran-style mantissa/exponent pair with mantissa in [0.1, 1)."""
    if value == 0.0:
        return 0.0, 0

    exponent = int(math.floor(math.log10(abs(value))) + 1)
    mantissa = value / (10.0 ** exponent)

    # Guard against roundoff pushing the mantissa out of range after formatting.
    rounded = float(f"{mantissa:.{digits}f}")
    if abs(rounded) >= 1.0:
        mantissa /= 10.0
        exponent += 1
    elif 0.0 < abs(rounded) < 0.1:
        mantissa *= 10.0
        exponent -= 1

    return mantissa, exponent


def format_fortran_real(value: float, *, width: int, digits: int, letter: str = "E") -> str:
    """Format a real number like Fortran ``Ew.d`` / ``Dw.d`` output."""
    if not math.isfinite(value):
        return f"{value:>{width}}"

    mantissa, exponent = _normalize_mantissa(float(value), digits)
    sign = "-" if math.copysign(1.0, value) < 0.0 and value != 0.0 else " "
    exp_sign = "+" if exponent >= 0 else "-"
    exp_digits = f"{abs(exponent):02d}"
    body = f"{abs(mantissa):.{digits}f}{letter}{exp_sign}{exp_digits}"
    return f"{sign}{body}".rjust(width)


def format_fortran_int(value: int, *, width: int) -> str:
    """Format an integer like Fortran ``Iw`` output."""
    return f"{int(value):>{width}d}"


def build_fortran_line(
    ints: Sequence[int] = (),
    *,
    int_width: int = 8,
    reals: Sequence[float] = (),
    real_width: int = 17,
    real_digits: int = 10,
    real_letter: str = "E",
) -> str:
    """Build a whitespace-prefixed Fortran-style record."""
    parts: list[str] = []
    for value in ints:
        parts.append(" " + format_fortran_int(value, width=int_width))
    for value in reals:
        parts.append(
            " " + format_fortran_real(
                float(value), width=real_width, digits=real_digits, letter=real_letter
            )
        )
    return "".join(parts)


def _write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_scalar_lines(path: Path, values: Sequence[float], *, fmt: str = "{: .16f}") -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(fmt.format(float(value)))
            handle.write("\n")


def _write_int_lines(path: Path, values: Sequence[int], *, width: int = 12) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(f"{int(value):>{width}d}\n")


@dataclass
class LegacyNeoWriter:
    """Writer for legacy ``xneo``-style auxiliary files."""

    extension: str | None
    progress: bool = False
    path_prefix: str = ""

    def _path(self, name: str) -> Path:
        return Path(f"{self.path_prefix}{name}")

    def _progress(self, name: str) -> None:
        if self.progress:
            print(f"write {name}")

    def prepare_run(self) -> None:
        """Reset per-run legacy files that Fortran recreates at startup."""
        if self.extension is not None:
            self._path(f"neolog.{self.extension}").write_text("", encoding="utf-8")

    def write_static_files(self, *, booz, grid: Mapping[str, object]) -> None:
        self._progress("dimension.dat")
        _write_lines(
            self._path("dimension.dat"),
            [
                f"{int(booz.rmnc.shape[0]):12d}",
                f"{int(booz.ixm.shape[0]):12d}",
                f"{int(booz.nfp):12d}",
                f"{int(len(grid['theta_arr'])):12d}",
                f"{int(len(grid['phi_arr'])):12d}",
                f"{0:12d}",
            ],
        )

        self._progress("es_arr.dat")
        es_lines = [
            "".join(
                format_fortran_real(val, width=18, digits=5, letter="D")
                for val in row
            )
            for row in zip(
                np.asarray(booz.es),
                np.asarray(booz.iota),
                np.asarray(booz.curr_pol),
                np.asarray(booz.curr_tor),
                np.zeros_like(np.asarray(booz.es))
                if getattr(booz, "pprime", None) is None
                else np.asarray(booz.pprime),
                np.zeros_like(np.asarray(booz.es))
                if getattr(booz, "sqrtg00", None) is None
                else np.asarray(booz.sqrtg00),
            )
        ]
        _write_lines(self._path("es_arr.dat"), es_lines)

        self._progress("mn_arr.dat")
        mn_lines = [
            f"{int(m):12d}{int(n):12d}"
            for m, n in zip(np.asarray(booz.ixm), np.asarray(booz.ixn))
        ]
        _write_lines(self._path("mn_arr.dat"), mn_lines)

        for name, arr in (
            ("rmnc_arr.dat", booz.rmnc),
            ("zmns_arr.dat", booz.zmns),
            ("lmns_arr.dat", booz.lmns),
            ("bmnc_arr.dat", booz.bmnc),
        ):
            self._progress(name)
            _write_scalar_lines(self._path(name), np.asarray(arr).reshape(-1))

        self._progress("theta_arr.dat")
        _write_scalar_lines(self._path("theta_arr.dat"), np.asarray(grid["theta_arr"]))
        self._progress("phi_arr.dat")
        _write_scalar_lines(self._path("phi_arr.dat"), np.asarray(grid["phi_arr"]))

    def write_surface_files(self, fields: Mapping[str, object]) -> None:
        for file_name, key in (
            ("b_s_arr.dat", "b"),
            ("r_s_arr.dat", "r"),
            ("z_s_arr.dat", "z"),
            ("l_s_arr.dat", "l"),
            ("isqrg_arr.dat", "isqrg"),
            ("sqrg11_arr.dat", "sqrg11"),
            ("kg_arr.dat", "kg"),
            ("pard_arr.dat", "pard"),
            ("r_tb_arr.dat", "r_tb"),
            ("z_tb_arr.dat", "z_tb"),
            ("p_tb_arr.dat", "p_tb"),
            ("b_tb_arr.dat", "b_tb"),
            ("r_pb_arr.dat", "r_pb"),
            ("z_pb_arr.dat", "z_pb"),
            ("p_pb_arr.dat", "p_pb"),
            ("b_pb_arr.dat", "b_pb"),
            ("gtbtb_arr.dat", "gtbtb"),
            ("gpbpb_arr.dat", "gpbpb"),
            ("gtbpb_arr.dat", "gtbpb"),
        ):
            self._progress(file_name)
            _write_scalar_lines(self._path(file_name), np.asarray(fields[key]).reshape(-1))

    def write_conver(self, history: Sequence[Sequence[float]]) -> None:
        self._progress("conver.dat")
        lines = [
            "".join(format_fortran_real(float(val), width=18, digits=5, letter="D") for val in row)
            for row in history
        ]
        _write_lines(self._path("conver.dat"), lines)

    def append_neolog(self, *, psi_ind: int, out: Mapping[str, object], epstot: float) -> None:
        name = f"neolog.{self.extension}" if self.extension else "neolog."
        line = build_fortran_line(
            ints=[
                int(psi_ind),
                int(out.get("n_iota", 0)),
                int(out.get("m_iota", 0)),
                int(out.get("n_gap", 0)),
                int(out.get("nfp_rat", 0)),
                int(out.get("nfl_rat", 0)) + 1,
                int(out.get("final_n", out.get("nintfp", 0))),
            ],
            int_width=6,
            reals=[float(epstot)],
            real_width=16,
            real_digits=8,
            real_letter="D",
        )
        path = self._path(name)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
