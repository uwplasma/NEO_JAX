"""JAX-friendly data models for NEO_JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import jax
import numpy as np

Array = jax.Array


def _split_static(
    obj: Any, static_fields: Iterable[str]
) -> Tuple[Tuple[Any, ...], Tuple[Tuple[str, Any], ...]]:
    static_fields = tuple(static_fields)
    children = []
    aux = []
    for name in obj.__dataclass_fields__.keys():
        value = getattr(obj, name)
        if name in static_fields:
            aux.append((name, value))
        else:
            children.append(value)
    return tuple(children), tuple(aux)


def _merge_static(
    cls: Any, children: Tuple[Any, ...], aux: Tuple[Tuple[str, Any], ...]
) -> Any:
    aux_map = dict(aux)
    values = []
    child_iter = iter(children)
    for name in cls.__dataclass_fields__.keys():
        if name in aux_map:
            values.append(aux_map[name])
        else:
            values.append(next(child_iter))
    return cls(*values)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class VmecData:
    """Container for vmec_jax outputs."""

    state: Any

    def tree_flatten(self):
        return _split_static(self, static_fields=())

    @classmethod
    def tree_unflatten(cls, aux, children):
        return _merge_static(cls, children, aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class BoozerData:
    """Boozer-coordinate data needed by NEO."""

    rmnc: Array | np.ndarray
    zmns: Array | np.ndarray
    lmns: Array | np.ndarray
    bmnc: Array | np.ndarray
    ixm: Array | np.ndarray
    ixn: Array | np.ndarray
    es: Array | np.ndarray
    iota: Array | np.ndarray
    curr_pol: Array | np.ndarray
    curr_tor: Array | np.ndarray
    nfp: int

    def tree_flatten(self):
        return _split_static(self, static_fields=("nfp",))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return _merge_static(cls, children, aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NeoInputs:
    """Inputs to the NEO solver."""

    config: Dict[str, Any]
    surfaces: Array

    def tree_flatten(self):
        return _split_static(self, static_fields=("config",))

    @classmethod
    def tree_unflatten(cls, aux, children):
        return _merge_static(cls, children, aux)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class NeoOutputs:
    """Outputs from the NEO solver."""

    eps_eff: Array
    eps_par: Array
    eps_tot: Array
    ctr_one: Array
    ctr_tot: Array
    diagnostics: Dict[str, Array]

    def tree_flatten(self):
        return _split_static(self, static_fields=())

    @classmethod
    def tree_unflatten(cls, aux, children):
        return _merge_static(cls, children, aux)
