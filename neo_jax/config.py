"""User-friendly configuration for NEO_JAX runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .control import ControlParams


@dataclass(frozen=True)
class NeoConfig:
    """High-level configuration for NEO_JAX runs.

    Surface selections may be specified as:
    - Integers (1-based NEO surface indices), or
    - Floats in [0, 1], interpreted as normalized toroidal flux ``s``.
    """

    surfaces: Sequence[int] | None = None
    theta_n: int = 64
    phi_n: int = 64
    max_m_mode: int = 0
    max_n_mode: int = 0
    npart: int = 40
    multra: int = 2
    acc_req: float = 0.02
    no_bins: int = 50
    nstep_per: int = 20
    nstep_min: int = 200
    nstep_max: int = 500
    calc_nstep_max: int = 0
    ref_swi: int = 2
    write_progress: bool = False
    write_diagnostic: bool = False

    @classmethod
    def from_control(cls, control: ControlParams) -> "NeoConfig":
        return cls(
            surfaces=control.fluxs_arr,
            theta_n=control.theta_n,
            phi_n=control.phi_n,
            max_m_mode=control.max_m_mode,
            max_n_mode=control.max_n_mode,
            npart=control.npart,
            multra=control.multra,
            acc_req=control.acc_req,
            no_bins=control.no_bins,
            nstep_per=control.nstep_per,
            nstep_min=control.nstep_min,
            nstep_max=control.nstep_max,
            calc_nstep_max=control.calc_nstep_max,
            ref_swi=control.ref_swi,
            write_progress=bool(control.write_progress),
            write_diagnostic=bool(control.write_diagnostic),
        )

    def to_control(self, *, in_file: str = "boozmn", out_file: str = "neo_out") -> ControlParams:
        """Convert to a ControlParams object (for CLI compatibility)."""
        return ControlParams(
            in_file=in_file,
            out_file=out_file,
            fluxs_arr=list(self.surfaces) if self.surfaces is not None else None,
            theta_n=self.theta_n,
            phi_n=self.phi_n,
            max_m_mode=self.max_m_mode,
            max_n_mode=self.max_n_mode,
            npart=self.npart,
            multra=self.multra,
            acc_req=self.acc_req,
            no_bins=self.no_bins,
            nstep_per=self.nstep_per,
            nstep_min=self.nstep_min,
            nstep_max=self.nstep_max,
            calc_nstep_max=self.calc_nstep_max,
            eout_swi=1,
            lab_swi=0,
            inp_swi=0,
            ref_swi=self.ref_swi,
            write_progress=int(self.write_progress),
            write_output_files=0,
            spline_test=0,
            write_integrate=0,
            write_diagnostic=int(self.write_diagnostic),
            calc_cur=0,
            cur_file="neo_cur",
            npart_cur=0,
            alpha_cur=0.0,
            write_cur_inte=0,
        )
