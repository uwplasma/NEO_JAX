"""User-facing API helpers for NEO_JAX."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import NeoConfig
from .control import ControlParams
from .data_models import BoozerData
from .driver import run_neo_from_boozer, run_neo_from_boozmn
from .io import booz_xform_to_boozerdata, read_boozmn, read_boozmn_metadata
from .results import NeoResults


def _control_from_config(config: NeoConfig, *, in_file: str = "boozmn", out_file: str = "neo_out") -> ControlParams:
    return config.to_control(in_file=in_file, out_file=out_file)


def _surface_s_from_index(index: int, ns_b: int) -> float:
    hs = 1.0 / (ns_b - 1)
    return (index - 1.5) * hs


def _resolve_surface_indices(
    surfaces: Sequence[int | float] | None,
    *,
    jlist: Sequence[int] | None,
    ns_b: int,
) -> list[int] | None:
    if surfaces is None:
        return None

    resolved: list[int] = []
    jlist_arr = list(jlist) if jlist is not None else None

    if jlist_arr is not None:
        s_vals = [_surface_s_from_index(idx, ns_b) for idx in jlist_arr]

    for val in surfaces:
        if isinstance(val, float) and 0.0 <= val <= 1.0:
            if jlist_arr is not None:
                best_idx = min(range(len(jlist_arr)), key=lambda i: abs(s_vals[i] - val))
                resolved.append(int(jlist_arr[best_idx]))
            else:
                idx = int(round(val * (ns_b - 1) + 1.5))
                idx = max(1, min(ns_b, idx))
                resolved.append(idx)
        else:
            resolved.append(int(val))

    return resolved


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
    surface_list = cfg.surfaces
    if surface_list is not None and any(isinstance(v, float) and 0.0 <= v <= 1.0 for v in surface_list):
        meta = read_boozmn_metadata(boozmn_path)
        resolved = _resolve_surface_indices(surface_list, jlist=meta["jlist"], ns_b=meta["ns_b"])
        cfg = replace(cfg, surfaces=resolved)
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
    surface_list = cfg.surfaces
    if surface_list is not None and any(isinstance(v, float) and 0.0 <= v <= 1.0 for v in surface_list):
        s_vals = list(np.asarray(booz.es))
        mapped = []
        for s_target in surface_list:
            if isinstance(s_target, float) and 0.0 <= s_target <= 1.0:
                best = min(range(len(s_vals)), key=lambda i: abs(s_vals[i] - s_target))
                mapped.append(best + 1)
            else:
                mapped.append(int(s_target))
        cfg = replace(cfg, surfaces=mapped)
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
    surface_list = cfg.surfaces
    if surface_list is not None and any(isinstance(v, float) and 0.0 <= v <= 1.0 for v in surface_list):
        jlist = None
        ns_b = None
        if isinstance(booz, dict):
            if "jlist" in booz:
                jlist = list(np.asarray(booz["jlist"]).astype(int))
            if "ns_b" in booz:
                ns_b = int(np.asarray(booz["ns_b"]).squeeze())
            elif "rmnc_b" in booz:
                ns_b = np.asarray(booz["rmnc_b"]).shape[0]
        else:
            if hasattr(booz, "jlist"):
                jlist = list(np.asarray(getattr(booz, "jlist")).astype(int))
            if hasattr(booz, "ns_b"):
                ns_b = int(np.asarray(getattr(booz, "ns_b")).squeeze())
            elif hasattr(booz, "rmnc_b"):
                ns_b = np.asarray(getattr(booz, "rmnc_b")).shape[0]
        if ns_b is None:
            raise ValueError("Unable to infer ns_b from booz_xform object")
        resolved = _resolve_surface_indices(surface_list, jlist=jlist, ns_b=ns_b)
        cfg = replace(cfg, surfaces=resolved)
    booz_data = booz_xform_to_boozerdata(
        booz,
        max_m_mode=cfg.max_m_mode if max_m_mode is None else max_m_mode,
        max_n_mode=cfg.max_n_mode if max_n_mode is None else max_n_mode,
        fluxs_arr=cfg.surfaces,
    )
    return run_boozer(booz_data, config=cfg, use_jax=use_jax, progress=progress)


def run_neo(
    source: BoozerData | str | Path | object,
    *,
    config: NeoConfig | None = None,
    surfaces: Sequence[int | float] | None = None,
    use_jax: bool = True,
    progress: bool | None = None,
    max_m_mode: int | None = None,
    max_n_mode: int | None = None,
) -> NeoResults:
    """Run NEO_JAX from a boozmn path, BoozerData, or booz_xform_jax-like object."""
    if isinstance(source, (str, Path)):
        return run_boozmn(
            source,
            config=config,
            surfaces=surfaces,
            use_jax=use_jax,
            progress=progress,
        )
    if isinstance(source, BoozerData):
        return run_boozer(
            source,
            config=config,
            surfaces=surfaces,
            use_jax=use_jax,
            progress=progress,
        )
    return run_booz_xform(
        source,
        config=config,
        surfaces=surfaces,
        use_jax=use_jax,
        progress=progress,
        max_m_mode=max_m_mode,
        max_n_mode=max_n_mode,
    )


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
