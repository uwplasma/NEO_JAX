"""Pipeline helpers for vmec_jax -> booz_xform_jax -> neo_jax."""

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
    """Run booz_xform_jax on an in-memory VMEC wout object.

    Parameters
    ----------
    wout:
        VMEC wout-like object (for example ``vmec_jax.WoutData``).
    mboz, nboz:
        Boozer resolution. If ``None``, defaults to VMEC mpol/ntor values.
    surfaces:
        Optional surface indices or s values in [0, 1]. If omitted, all
        VMEC half-grid surfaces are used.
    flux:
        If ``True``, attempt to load flux profile arrays from ``wout``.
    jit:
        If ``True``, jit-compile the Boozer transform.
    """
    try:
        from booz_xform_jax import Booz_xform
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "booz_xform_jax is required for vmec -> booz pipeline. "
            "Install booz_xform_jax or add it to PYTHONPATH."
        ) from exc

    bx = Booz_xform()
    bx.read_wout_data(wout, flux=flux)
    if mboz is not None:
        bx.mboz = int(mboz)
    if nboz is not None:
        bx.nboz = int(nboz)
    if surfaces is not None:
        bx.register_surfaces(surfaces)
    return bx.run_jax(jit=jit)


def booz_xform_from_vmec_state_jax(
    *,
    vmec_run: Any,
    mboz: int | None = None,
    nboz: int | None = None,
    surfaces: Sequence[int | float] | None = None,
    jit: bool = True,
) -> Mapping[str, Any]:
    """JAX-native VMEC state -> Boozer transform using booz_xform_jax."""
    try:
        import jax
        from booz_xform_jax.jax_api import prepare_booz_xform_constants, booz_xform_jax_impl
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "booz_xform_jax is required for the JAX-native VMEC -> Boozer path."
        ) from exc

    try:
        from vmec_jax.booz_input import booz_xform_inputs_from_state
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "vmec_jax with booz_input is required for the JAX-native VMEC -> Boozer path."
        ) from exc

    inputs = booz_xform_inputs_from_state(
        state=vmec_run.state,
        static=vmec_run.static,
        indata=vmec_run.indata,
        signgs=int(vmec_run.signgs),
    )

    mboz_val = int(mboz) if mboz is not None else int(np.max(np.asarray(inputs.xm))) + 1
    nboz_val = int(nboz) if nboz is not None else int(np.max(np.abs(np.asarray(inputs.xn)))) // int(inputs.nfp)

    constants, grids = prepare_booz_xform_constants(
        nfp=int(inputs.nfp),
        mboz=mboz_val,
        nboz=nboz_val,
        asym=bool(vmec_run.static.cfg.lasym),
        xm=np.asarray(inputs.xm),
        xn=np.asarray(inputs.xn),
        xm_nyq=np.asarray(inputs.xm_nyq),
        xn_nyq=np.asarray(inputs.xn_nyq),
    )

    ns_b_full = int(inputs.rmnc.shape[0])
    s_half_full = jax.numpy.asarray(0.5 * (vmec_run.static.s[:-1] + vmec_run.static.s[1:]))
    if surfaces is None:
        surface_indices = None
        s_selected = s_half_full
    else:
        s_vals = list(np.asarray(s_half_full))
        surface_indices_list = []
        for val in surfaces:
            if isinstance(val, float) and 0.0 <= val <= 1.0:
                best = min(range(ns_b_full), key=lambda i: abs(s_vals[i] - val))
                surface_indices_list.append(best)
            else:
                surface_indices_list.append(int(val) - 1)
        surface_indices = jax.numpy.asarray(surface_indices_list, dtype=jax.numpy.int32)
        s_selected = jax.numpy.take(s_half_full, surface_indices, axis=0)

    booz_fn = booz_xform_jax_impl
    if jit:
        booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))

    booz_out = booz_fn(
        rmnc=inputs.rmnc,
        zmns=inputs.zmns,
        lmns=inputs.lmns,
        bmnc=inputs.bmnc,
        bsubumnc=inputs.bsubumnc,
        bsubvmnc=inputs.bsubvmnc,
        iota=inputs.iota,
        xm=inputs.xm,
        xn=inputs.xn,
        xm_nyq=inputs.xm_nyq,
        xn_nyq=inputs.xn_nyq,
        constants=constants,
        grids=grids,
        bmns=inputs.bmns,
        bsubumns=inputs.bsubumns,
        bsubvmns=inputs.bsubvmns,
        surface_indices=surface_indices,
    )
    if "s_b" not in booz_out:
        booz_out["s_b"] = s_selected
    if "ns_b" not in booz_out:
        booz_out["ns_b"] = jax.numpy.asarray(ns_b_full)
    if surface_indices is not None and "jlist" not in booz_out:
        booz_out["jlist"] = surface_indices + 1
    return booz_out


def _resolve_vmec_wout(
    vmec_source: Any,
    *,
    vmec_kwargs: dict | None = None,
    fast_bcovar: bool = True,
) -> Any:
    vmec_kwargs = vmec_kwargs or {}
    if isinstance(vmec_source, (str, Path)):
        try:
            import vmec_jax as vj
            from vmec_jax.driver import run_fixed_boundary, wout_from_fixed_boundary_run
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "vmec_jax is required for vmec -> booz pipeline. "
                "Install vmec_jax or add it to PYTHONPATH."
            ) from exc
        run = run_fixed_boundary(vmec_source, **vmec_kwargs)
        return wout_from_fixed_boundary_run(run, include_fsq=False, fast_bcovar=fast_bcovar)

    if hasattr(vmec_source, "state") and hasattr(vmec_source, "static"):
        try:
            from vmec_jax.driver import wout_from_fixed_boundary_run
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "vmec_jax is required for vmec -> booz pipeline. "
                "Install vmec_jax or add it to PYTHONPATH."
            ) from exc
        return wout_from_fixed_boundary_run(vmec_source, include_fsq=False, fast_bcovar=fast_bcovar)

    if hasattr(vmec_source, "rmnc"):
        return vmec_source

    raise TypeError(
        "vmec_source must be a vmec_jax FixedBoundaryRun, WoutData, or input path"
    )


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
    """Run vmec_jax -> booz_xform_jax -> neo_jax in one workflow.

    This requires a JAX-native ``booz_xform_fn`` (for example from
    ``booz_xform_jax.jax_api``). ``vmec_source`` may be a
    ``vmec_jax.FixedBoundaryRun``, a ``vmec_jax.WoutData`` object, or a path to
    a VMEC input file.
    """
    wout = _resolve_vmec_wout(vmec_source, vmec_kwargs=vmec_kwargs, fast_bcovar=fast_bcovar)

    booz_kwargs = booz_kwargs or {}
    if booz_xform_fn is None:
        booz_output = booz_xform_from_vmec_wout(wout, **booz_kwargs)
    else:
        booz_output = booz_xform_fn(wout, **booz_kwargs)
    return run_neo(booz_output, config=neo_config, use_jax=use_jax, progress=progress)


def run_vmec_boozer_neo_jax(
    vmec_run: Any,
    *,
    booz_kwargs: dict | None = None,
    neo_config: NeoConfig | None = None,
    jax_surface_scan: bool = True,
    progress: bool | None = None,
) -> Any:
    """JAX-native VMEC -> Boozer -> NEO pipeline using the JAX surface scan."""
    booz_kwargs = booz_kwargs or {}
    booz_output = booz_xform_from_vmec_state_jax(vmec_run=vmec_run, **booz_kwargs)
    return run_neo(
        booz_output,
        config=neo_config,
        use_jax=True,
        progress=progress,
        jax_surface_scan=jax_surface_scan,
    )


def build_vmec_boozer_neo_jax(
    vmec_run: Any,
    *,
    booz_kwargs: dict | None = None,
    neo_config: NeoConfig | None = None,
    jit: bool = True,
):
    """Return a callable `solve(state)` for the JAX-native VMEC→Boozer→NEO path.

    This precomputes Boozer constants and surface selections so the returned
    function is suitable for repeated calls (and optional JIT).
    """
    booz_kwargs = booz_kwargs or {}
    cfg = neo_config or NeoConfig()

    try:
        import jax
        import jax.numpy as jnp
        from booz_xform_jax.jax_api import prepare_booz_xform_constants, booz_xform_jax_impl
    except ImportError as exc:  # pragma: no cover
        raise ImportError("booz_xform_jax is required for build_vmec_boozer_neo_jax") from exc

    try:
        from vmec_jax.booz_input import booz_xform_inputs_from_state
        from vmec_jax.energy import flux_profiles_from_indata
        from vmec_jax.profiles import eval_profiles
        from vmec_jax.vmec_tomnsp import vmec_trig_tables
    except ImportError as exc:  # pragma: no cover
        raise ImportError("vmec_jax with booz_input is required for build_vmec_boozer_neo_jax") from exc

    from .driver import run_neo_from_boozer_jax
    from .io import booz_xform_to_boozerdata_jax

    # Precompute static Boozer constants from the current state.
    inputs0 = booz_xform_inputs_from_state(
        state=vmec_run.state,
        static=vmec_run.static,
        indata=vmec_run.indata,
        signgs=int(vmec_run.signgs),
    )

    nyq_m = np.asarray(inputs0.xm_nyq)
    nyq_n = np.asarray(inputs0.xn_nyq)
    nfp_int = int(inputs0.nfp)
    mmax = int(np.max(nyq_m)) if nyq_m.size else 0
    nmax = int(np.max(np.abs(nyq_n))) // nfp_int if nyq_n.size else 0
    trig = vmec_trig_tables(
        ntheta=int(vmec_run.static.cfg.ntheta),
        nzeta=int(vmec_run.static.cfg.nzeta),
        nfp=nfp_int,
        mmax=mmax,
        nmax=nmax,
        lasym=bool(vmec_run.static.cfg.lasym),
        dtype=np.asarray(inputs0.rmnc).dtype,
        cache=True,
    )

    mboz_val = int(booz_kwargs.get("mboz") or (np.max(np.asarray(inputs0.xm)) + 1))
    nboz_val = int(
        booz_kwargs.get("nboz")
        or (np.max(np.abs(np.asarray(inputs0.xn))) // int(inputs0.nfp))
    )

    constants, grids = prepare_booz_xform_constants(
        nfp=int(inputs0.nfp),
        mboz=mboz_val,
        nboz=nboz_val,
        asym=bool(vmec_run.static.cfg.lasym),
        xm=np.asarray(inputs0.xm),
        xn=np.asarray(inputs0.xn),
        xm_nyq=np.asarray(inputs0.xm_nyq),
        xn_nyq=np.asarray(inputs0.xn_nyq),
    )

    ns_full = int(inputs0.rmnc.shape[0])
    s_half_full = jnp.asarray(0.5 * (vmec_run.static.s[:-1] + vmec_run.static.s[1:]))
    s_half_eval = jnp.concatenate([vmec_run.static.s[:1], s_half_full], axis=0)
    profiles_half = eval_profiles(vmec_run.indata, s_half_eval)
    flux = flux_profiles_from_indata(vmec_run.indata, vmec_run.static.s, signgs=int(vmec_run.signgs))
    if cfg.surfaces is None:
        surface_indices = None
        s_selected = s_half_full
    else:
        s_vals = list(np.asarray(s_half_full))
        surface_indices_list = []
        for val in cfg.surfaces:
            if isinstance(val, float) and 0.0 <= val <= 1.0:
                best = min(range(ns_full), key=lambda i: abs(s_vals[i] - val))
                surface_indices_list.append(best)
            else:
                surface_indices_list.append(int(val) - 1)
        surface_indices = jnp.asarray(surface_indices_list, dtype=jnp.int32)
        s_selected = jnp.take(s_half_full, surface_indices, axis=0)

    control = cfg.to_control()
    # Modes already filtered in the JAX pipeline; skip re-masking in Fourier sums.
    control = replace(control, max_m_mode=-1, max_n_mode=-1)

    # Precompute mode indices for JIT-safe slicing.
    xm_b_np = np.asarray(grids.xm_b)
    xn_b_np = np.asarray(grids.xn_b)
    max_m = int(cfg.max_m_mode) if cfg.max_m_mode > 0 else int(np.max(np.abs(xm_b_np)))
    max_n = int(cfg.max_n_mode) if cfg.max_n_mode > 0 else int(np.max(np.abs(xn_b_np)))
    mode_indices = np.where((np.abs(xm_b_np) <= max_m) & (np.abs(xn_b_np) <= max_n))[0]

    def _solve(state):
        inputs = booz_xform_inputs_from_state(
            state=state,
            static=vmec_run.static,
            indata=vmec_run.indata,
            signgs=int(vmec_run.signgs),
            trig=trig,
            flux=flux,
            profiles_half=profiles_half,
        )
        booz_out = booz_xform_jax_impl(
            rmnc=inputs.rmnc,
            zmns=inputs.zmns,
            lmns=inputs.lmns,
            bmnc=inputs.bmnc,
            bsubumnc=inputs.bsubumnc,
            bsubvmnc=inputs.bsubvmnc,
            iota=inputs.iota,
            xm=inputs.xm,
            xn=inputs.xn,
            xm_nyq=inputs.xm_nyq,
            xn_nyq=inputs.xn_nyq,
            constants=constants,
            grids=grids,
            bmns=inputs.bmns,
            bsubumns=inputs.bsubumns,
            bsubvmns=inputs.bsubvmns,
            surface_indices=surface_indices,
        )
        booz_out["s_b"] = s_selected
        booz_out["ns_b"] = ns_full
        if surface_indices is not None:
            booz_out["jlist"] = surface_indices + 1
        booz = booz_xform_to_boozerdata_jax(
            booz_out,
            max_m_mode=cfg.max_m_mode,
            max_n_mode=cfg.max_n_mode,
            nfp_override=int(inputs0.nfp),
            mode_indices=mode_indices,
        )
        return run_neo_from_boozer_jax(booz, control, skip_fourier_mask=True)

    if jit:
        _solve = jax.jit(_solve)
    return _solve
