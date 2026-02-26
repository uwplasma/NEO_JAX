"""Control file parsing for NEO_JAX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ControlParams:
    in_file: str
    out_file: str
    fluxs_arr: Optional[List[int]]
    theta_n: int
    phi_n: int
    max_m_mode: int
    max_n_mode: int
    npart: int
    multra: int
    acc_req: float
    no_bins: int
    nstep_per: int
    nstep_min: int
    nstep_max: int
    calc_nstep_max: int
    eout_swi: int
    lab_swi: int
    inp_swi: int
    ref_swi: int
    write_progress: int
    write_output_files: int
    spline_test: int
    write_integrate: int
    write_diagnostic: int
    calc_cur: int
    cur_file: str
    npart_cur: int
    alpha_cur: float
    write_cur_inte: int


def _read_lines(path: Path) -> List[str]:
    lines: List[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line:
            lines.append(line)
    return lines


def _parse_lines(lines: List[str]) -> ControlParams:
    idx = 0

    in_file = lines[idx]
    idx += 1
    out_file = lines[idx]
    idx += 1
    no_fluxs = int(lines[idx])
    idx += 1

    fluxs_arr: Optional[List[int]]
    if no_fluxs <= 0:
        if idx < len(lines):
            idx += 1
        fluxs_arr = None
    else:
        fluxs_arr = [int(v) for v in lines[idx].split()]
        idx += 1

    theta_n = int(lines[idx]); idx += 1
    phi_n = int(lines[idx]); idx += 1
    max_m_mode = int(lines[idx]); idx += 1
    max_n_mode = int(lines[idx]); idx += 1
    npart = int(lines[idx]); idx += 1
    multra = int(lines[idx]); idx += 1
    acc_req = float(lines[idx]); idx += 1
    no_bins = int(lines[idx]); idx += 1
    nstep_per = int(lines[idx]); idx += 1
    nstep_min = int(lines[idx]); idx += 1
    nstep_max = int(lines[idx]); idx += 1
    calc_nstep_max = int(lines[idx]); idx += 1
    eout_swi = int(lines[idx]); idx += 1
    lab_swi = int(lines[idx]); idx += 1
    inp_swi = int(lines[idx]); idx += 1
    ref_swi = int(lines[idx]); idx += 1
    write_progress = int(lines[idx]); idx += 1
    write_output_files = int(lines[idx]); idx += 1
    spline_test = int(lines[idx]); idx += 1
    write_integrate = int(lines[idx]); idx += 1
    write_diagnostic = int(lines[idx]); idx += 1

    idx += 3

    calc_cur = int(lines[idx]); idx += 1
    cur_file = lines[idx]; idx += 1
    npart_cur = int(lines[idx]); idx += 1
    alpha_cur = float(lines[idx]); idx += 1
    write_cur_inte = int(lines[idx]); idx += 1

    return ControlParams(
        in_file=in_file,
        out_file=out_file,
        fluxs_arr=fluxs_arr,
        theta_n=theta_n,
        phi_n=phi_n,
        max_m_mode=max_m_mode,
        max_n_mode=max_n_mode,
        npart=npart,
        multra=multra,
        acc_req=acc_req,
        no_bins=no_bins,
        nstep_per=nstep_per,
        nstep_min=nstep_min,
        nstep_max=nstep_max,
        calc_nstep_max=calc_nstep_max,
        eout_swi=eout_swi,
        lab_swi=lab_swi,
        inp_swi=inp_swi,
        ref_swi=ref_swi,
        write_progress=write_progress,
        write_output_files=write_output_files,
        spline_test=spline_test,
        write_integrate=write_integrate,
        write_diagnostic=write_diagnostic,
        calc_cur=calc_cur,
        cur_file=cur_file,
        npart_cur=npart_cur,
        alpha_cur=alpha_cur,
        write_cur_inte=write_cur_inte,
    )


def read_control(path: str | Path) -> ControlParams:
    lines = _read_lines(Path(path))
    last_err: Exception | None = None
    for offset in range(0, 6):
        try:
            return _parse_lines(lines[offset:])
        except (ValueError, IndexError) as exc:
            last_err = exc
            continue
    raise ValueError(f"Failed to parse control file {path}") from last_err
