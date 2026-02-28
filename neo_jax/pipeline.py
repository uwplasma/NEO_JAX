"""Pipeline helpers for vmec_jax -> booz_xform_jax -> neo_jax."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

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

    This requires a JAX-native `booz_xform_fn` (e.g., from booz_xform_jax.jax_api)
    and a vmec_source that is either:
      - a vmec_jax FixedBoundaryRun,
      - a vmec_jax WoutData object,
      - or a path to a VMEC input file.
    """
    wout = _resolve_vmec_wout(vmec_source, vmec_kwargs=vmec_kwargs, fast_bcovar=fast_bcovar)

    booz_kwargs = booz_kwargs or {}
    if booz_xform_fn is None:
        booz_output = booz_xform_from_vmec_wout(wout, **booz_kwargs)
    else:
        booz_output = booz_xform_fn(wout, **booz_kwargs)
    return run_neo(booz_output, config=neo_config, use_jax=use_jax, progress=progress)
