"""VMEX -> booz_xform_jax -> NEO_JAX pipelines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from dataclasses import replace

import numpy as np

from .api import run_neo
from .config import NeoConfig


def run_boozer_to_neo(
    booz_output: Mapping[str, Any],
    *,
    config: NeoConfig | None = None,
    use_jax: bool = True,
    progress: bool | None = None,
) -> Any:
    """Run NEO_JAX from a booz_xform_jax output mapping."""
    return run_neo(booz_output, config=config, use_jax=use_jax, progress=progress)


def booz_xform_from_vmec_wout(
    wout: Any,
    *,
    mboz: int | None = None,
    nboz: int | None = None,
    surfaces: Sequence[int | float] | None = None,
    flux: bool = False,
    jit: bool = True,
) -> Mapping[str, Any]:
    """Transform a symmetric VMEC wout; integer surfaces are 1-based half-grid rows."""
    from booz_xform_jax import Booz_xform

    if bool(getattr(wout, "lasym", getattr(wout, "asym", False))):
        raise ValueError("NEO_JAX requires stellarator-symmetric equilibria")
    bx = Booz_xform(verbose=0)
    bx.read_wout_data(wout, flux=flux)
    if mboz is not None:
        bx.mboz = int(mboz)
    if nboz is not None:
        bx.nboz = int(nboz)
    indices = _surface_indices(np.asarray(bx.s_in), surfaces)
    bx.compute_surfs = indices.tolist()
    out = bx.run_jax(jit=jit)
    out["s_b"] = np.asarray(bx.s_in)[indices]
    out["ns_b"] = len(bx.s_in) + 1
    return out


def _surface_indices(s_half, surfaces):
    if surfaces is None:
        return np.arange(len(s_half), dtype=int)
    indices = []
    for val in surfaces:
        if isinstance(val, (float, np.floating)):
            if not np.isfinite(val) or not 0 <= val <= 1:
                raise ValueError("Flux surfaces must lie in [0, 1]")
            index = int(np.argmin(np.abs(s_half - val)))
        else:
            index = int(val) - 1
        if not 0 <= index < len(s_half):
            raise ValueError("Surface index is outside the VMEC half grid")
        indices.append(index)
    if not indices:
        raise ValueError("At least one surface is required")
    return np.asarray(indices, dtype=int)


def _build_boozer_transform(vmec_run, *, mboz=None, nboz=None, surfaces=None):
    import jax.numpy as jnp
    from booz_xform_jax.jax_api import booz_xform_jax_impl, prepare_booz_xform_constants
    from vmex.core.boozer_tables import boozer_input_tables

    rt = vmec_run.runtime
    if rt.setup.lasym:
        raise ValueError("NEO_JAX requires stellarator-symmetric equilibria")
    s_full = np.asarray(rt.setup.s_full)
    s_half = 0.5 * (s_full[:-1] + s_full[1:])
    indices = _surface_indices(s_half, surfaces)
    first = boozer_input_tables(vmec_run.state, rt, int(indices[0]) + 1)
    xm, xn = np.asarray(first["xm"]), np.asarray(first["xn"])
    constants, grids = prepare_booz_xform_constants(
        nfp=int(rt.resolution.nfp),
        mboz=int(rt.resolution.mpol if mboz is None else mboz),
        nboz=int(rt.resolution.ntor if nboz is None else nboz),
        asym=False, xm=xm, xn=xn, xm_nyq=xm, xn_nyq=xn,
    )

    def transform(state):
        tables = [boozer_input_tables(state, rt, int(i) + 1) for i in indices]
        inputs = {name: jnp.stack([table[name] for table in tables]) for name in
                  ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")}
        out = booz_xform_jax_impl(
            **inputs, xm=jnp.asarray(xm), xn=jnp.asarray(xn),
            xm_nyq=jnp.asarray(xm), xn_nyq=jnp.asarray(xn), constants=constants, grids=grids,
        )
        out.update(s_b=jnp.asarray(s_half[indices]), ns_b=len(s_full),
                   jlist=jnp.asarray(indices + 2))
        return out

    return transform, grids, int(rt.resolution.nfp)


def booz_xform_from_vmec_state_jax(
    *,
    vmec_run: Any,
    mboz: int | None = None,
    nboz: int | None = None,
    surfaces: Sequence[int | float] | None = None,
    jit: bool = True,
) -> Mapping[str, Any]:
    """Transform a VMEX Equilibrium state without interrupting its derivatives."""
    import jax

    transform, _, _ = _build_boozer_transform(
        vmec_run, mboz=mboz, nboz=nboz, surfaces=surfaces)
    return (jax.jit(transform) if jit else transform)(vmec_run.state)


def _resolve_vmec_wout(
    vmec_source: Any,
    *,
    vmec_kwargs: dict | None = None,
) -> Any:
    if hasattr(vmec_source, "rmnc"):
        return vmec_source
    if hasattr(vmec_source, "runtime") and hasattr(vmec_source, "state"):
        return vmec_source.wout

    from vmex import VmecInput, read_wout
    from vmex import optimize

    if isinstance(vmec_source, (str, Path)):
        if Path(vmec_source).suffix == ".nc":
            return read_wout(vmec_source)
        vmec_source = VmecInput.from_file(vmec_source)
    if isinstance(vmec_source, VmecInput):
        return optimize.solve_equilibrium(vmec_source, **(vmec_kwargs or {})).wout
    raise TypeError("vmec_source must be a VMEX Equilibrium, VmecInput, wout, or file path")


def run_vmec_boozer_neo(
    vmec_source: Any,
    *,
    booz_xform_fn: Callable[..., Mapping[str, Any]] | None = None,
    booz_kwargs: dict | None = None,
    vmec_kwargs: dict | None = None,
    neo_config: NeoConfig | None = None,
    use_jax: bool = True,
    progress: bool | None = None,
    fast_bcovar: bool = True,
) -> Any:
    """Solve a VMEX input or transform a wout, then evaluate effective ripple."""
    wout = _resolve_vmec_wout(vmec_source, vmec_kwargs=vmec_kwargs)
    cfg = neo_config or NeoConfig()
    kwargs = dict(booz_kwargs or {})
    if booz_xform_fn is None:
        kwargs.setdefault("surfaces", cfg.surfaces)
        booz_output = booz_xform_from_vmec_wout(wout, **kwargs)
        cfg = replace(cfg, surfaces=None)
    else:
        booz_output = booz_xform_fn(wout, **kwargs)
    return run_neo(booz_output, config=cfg, use_jax=use_jax, progress=progress)


def run_vmec_boozer_neo_jax(
    vmec_run: Any,
    *,
    booz_kwargs: dict | None = None,
    neo_config: NeoConfig | None = None,
    jax_surface_scan: bool = True,
    progress: bool | None = None,
) -> Any:
    """JAX-native VMEC -> Boozer -> NEO pipeline using the JAX surface scan."""
    kwargs = dict(booz_kwargs or {})
    cfg = neo_config or NeoConfig()
    if jax_surface_scan:
        jit = kwargs.pop("jit", True)
        return build_vmec_boozer_neo_jax(
            vmec_run, booz_kwargs=kwargs, neo_config=cfg, jit=jit)(vmec_run.state)
    kwargs.setdefault("surfaces", cfg.surfaces)
    booz_output = booz_xform_from_vmec_state_jax(vmec_run=vmec_run, **kwargs)
    return run_neo(booz_output, config=replace(cfg, surfaces=None), progress=progress)


def build_vmec_boozer_neo_jax(
    vmec_run: Any,
    *,
    booz_kwargs: dict | None = None,
    neo_config: NeoConfig | None = None,
    jit: bool = True,
):
    """Build a reusable differentiable VMEX state -> full Boozer geometry -> NEO solve."""
    import jax
    from .driver import run_neo_from_boozer_jax, _resolve_max_rational_field_periods
    from .io import booz_xform_to_boozerdata_jax

    cfg = neo_config or NeoConfig()
    kwargs = dict(booz_kwargs or {})
    kwargs.pop("jit", None)
    kwargs.setdefault("surfaces", cfg.surfaces)
    transform, grids, nfp = _build_boozer_transform(vmec_run, **kwargs)
    rt = vmec_run.runtime
    work_guard = None
    # Prescribed iota is independent of the state and safe for a static work guard.
    if int(rt.setup.ncurr) == 0:
        s_full = np.asarray(rt.setup.s_full)
        s_half = 0.5 * (s_full[:-1] + s_full[1:])
        indices = _surface_indices(s_half, kwargs["surfaces"])
        work_guard = (np.asarray(rt.setup.iotas)[indices + 1], s_half[indices])
    elif jit and _resolve_max_rational_field_periods(cfg.max_rational_field_periods) is not None:
        raise ValueError("Compiled NCURR=1 requires max_rational_field_periods=0")
    xm, xn = np.asarray(grids.xm_b), np.asarray(grids.xn_b)
    mode_indices = np.flatnonzero(
        (np.abs(xm) <= (cfg.max_m_mode if cfg.max_m_mode > 0 else np.max(np.abs(xm)))) &
        (np.abs(xn) <= (cfg.max_n_mode if cfg.max_n_mode > 0 else np.max(np.abs(xn)))))
    control = replace(cfg.to_control(), fluxs_arr=None, max_m_mode=-1, max_n_mode=-1)

    def solve(state):
        booz = booz_xform_to_boozerdata_jax(
            transform(state), nfp_override=nfp, mode_indices=mode_indices)
        return run_neo_from_boozer_jax(
            booz, control, skip_fourier_mask=True,
            max_rational_field_periods=cfg.max_rational_field_periods,
            rational_surface_policy=cfg.rational_surface_policy, _work_guard=work_guard,
            sequential=cfg.sequential,
        )

    return jax.jit(solve) if jit else solve
