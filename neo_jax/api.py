"""User-facing API helpers for NEO_JAX."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from .config import NeoConfig
from .control import ControlParams
from .data_models import BoozerData
from .driver import run_neo_from_boozer, run_neo_from_boozmn
from .io import booz_xform_to_boozerdata, read_boozmn
from .results import NeoResults


def _control_from_config(config: NeoConfig, *, in_file: str = "boozmn", out_file: str = "neo_out") -> ControlParams:
    return config.to_control(in_file=in_file, out_file=out_file)


def run_boozmn(
    boozmn_path: str | Path,
    *,
    config: NeoConfig | None = None,
    surfaces: Sequence[int] | None = None,
    use_jax: bool = True,
    progress: bool | None = None,
) -> NeoResults:
    """Run NEO_JAX from a boozmn file using a simplified configuration."""
    cfg = config or NeoConfig()
    if surfaces is not None:
        cfg = replace(cfg, surfaces=list(surfaces))
    ctrl = _control_from_config(cfg)
    if progress is None:
        progress = cfg.write_progress
    return run_neo_from_boozmn(str(boozmn_path), ctrl, use_jax=use_jax, progress=progress)


def run_boozer(
    booz: BoozerData,
    *,
    config: NeoConfig | None = None,
    surfaces: Sequence[int] | None = None,
    use_jax: bool = True,
    progress: bool | None = None,
) -> NeoResults:
    """Run NEO_JAX from a BoozerData object (e.g., booz_xform_jax output)."""
    cfg = config or NeoConfig()
    if surfaces is not None:
        cfg = replace(cfg, surfaces=list(surfaces))
    ctrl = _control_from_config(cfg)
    if progress is None:
        progress = cfg.write_progress
    return run_neo_from_boozer(booz, ctrl, use_jax=use_jax, progress=progress)


def run_booz_xform(
    booz: object,
    *,
    config: NeoConfig | None = None,
    surfaces: Sequence[int] | None = None,
    use_jax: bool = True,
    progress: bool | None = None,
    max_m_mode: int | None = None,
    max_n_mode: int | None = None,
) -> NeoResults:
    """Run NEO_JAX from a booz_xform_jax-style object or mapping."""
    cfg = config or NeoConfig()
    if surfaces is not None:
        cfg = replace(cfg, surfaces=list(surfaces))
    booz_data = booz_xform_to_boozerdata(
        booz,
        max_m_mode=cfg.max_m_mode if max_m_mode is None else max_m_mode,
        max_n_mode=cfg.max_n_mode if max_n_mode is None else max_n_mode,
        fluxs_arr=cfg.surfaces,
    )
    return run_boozer(booz_data, config=cfg, use_jax=use_jax, progress=progress)


def load_boozmn(
    boozmn_path: str | Path,
    *,
    max_m_mode: int = 0,
    max_n_mode: int = 0,
    surfaces: Sequence[int] | None = None,
) -> BoozerData:
    """Load a boozmn file into BoozerData for custom workflows."""
    return read_boozmn(
        str(boozmn_path),
        max_m_mode=max_m_mode,
        max_n_mode=max_n_mode,
        fluxs_arr=surfaces,
    )
