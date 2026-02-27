"""Pipeline helpers for vmec_jax -> booz_xform_jax -> neo_jax."""

from __future__ import annotations

from typing import Any, Callable, Mapping

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


def run_vmec_boozer_neo(
    vmec_state: Any,
    *,
    booz_xform_fn: Callable[..., Mapping[str, Any]] | None = None,
    booz_kwargs: dict | None = None,
    neo_config: NeoConfig | None = None,
    use_jax: bool = True,
    progress: bool | None = None,
) -> Any:
    """Run vmec_jax -> booz_xform_jax -> neo_jax in one workflow.

    This requires a JAX-native `booz_xform_fn` (e.g., from booz_xform_jax.jax_api)
    and a vmec_state that exposes the arrays needed by the transform.
    """
    if booz_xform_fn is None:
        raise ValueError("booz_xform_fn must be provided for vmec -> booz -> neo pipeline")
    booz_kwargs = booz_kwargs or {}
    booz_output = booz_xform_fn(vmec_state, **booz_kwargs)
    return run_neo(booz_output, config=neo_config, use_jax=use_jax, progress=progress)
