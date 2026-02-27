"""User-friendly result containers for NEO_JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


_ALIAS_MAP = {
    "flux_index": "flux_index",
    "surface_index": "flux_index",
    "psi": "s",
    "s": "s",
    "sqrt_s": "sqrt_s",
    "reff": "r_eff",
    "r_eff": "r_eff",
    "iota": "iota",
    "b_ref": "b_ref",
    "bref": "b_ref",
    "r_ref": "r_ref",
    "rref": "r_ref",
    "epstot": "epsilon_effective",
    "epsilon_effective": "epsilon_effective",
    "eps_eff": "epsilon_effective",
    "epspar": "epsilon_effective_by_class",
    "epsilon_effective_by_class": "epsilon_effective_by_class",
    "ctrone": "ctrone",
    "ctrtot": "ctrtot",
    "bareph": "bareph",
    "barept": "barept",
    "yps": "yps",
    "diagnostics": "diagnostics",
}


@dataclass(frozen=True)
class NeoSurfaceResult:
    """Results for a single flux surface."""

    flux_index: int
    s: float
    r_eff: float
    iota: float
    b_ref: float
    r_ref: float
    epsilon_effective: float
    epsilon_effective_by_class: np.ndarray
    ctrone: float
    ctrtot: float
    bareph: float
    barept: float
    yps: float
    diagnostics: Mapping[str, object]

    @property
    def epstot(self) -> float:
        return self.epsilon_effective

    @property
    def sqrt_s(self) -> float:
        return float(np.sqrt(max(self.s, 0.0)))

    @property
    def reff(self) -> float:
        return self.r_eff

    @property
    def psi(self) -> float:
        return self.s

    @property
    def epspar(self) -> np.ndarray:
        return self.epsilon_effective_by_class

    def __getitem__(self, key: str):
        mapped = _ALIAS_MAP.get(key)
        if mapped is None:
            raise KeyError(key)
        return getattr(self, mapped)

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def to_dict(self) -> dict:
        return {
            "flux_index": self.flux_index,
            "s": self.s,
            "psi": self.s,
            "sqrt_s": self.sqrt_s,
            "r_eff": self.r_eff,
            "reff": self.r_eff,
            "iota": self.iota,
            "b_ref": self.b_ref,
            "r_ref": self.r_ref,
            "epstot": self.epsilon_effective,
            "epspar": self.epsilon_effective_by_class,
            "ctrone": self.ctrone,
            "ctrtot": self.ctrtot,
            "bareph": self.bareph,
            "barept": self.barept,
            "yps": self.yps,
            "diagnostics": self.diagnostics,
        }


class NeoResults(Sequence[NeoSurfaceResult]):
    """Container for multiple surface results with convenience accessors."""

    def __init__(self, results: Iterable[NeoSurfaceResult]):
        self._results = tuple(results)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._results)

    def __iter__(self) -> Iterator[NeoSurfaceResult]:  # pragma: no cover - trivial
        return iter(self._results)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._collect(key)
        return self._results[key]

    def __getattr__(self, name: str):
        if name in _ALIAS_MAP:
            return self._collect(name)
        raise AttributeError(name)

    def _collect(self, key: str) -> np.ndarray:
        mapped = _ALIAS_MAP.get(key)
        if mapped is None:
            raise KeyError(key)
        if mapped == "epsilon_effective_by_class":
            return np.stack([res.epsilon_effective_by_class for res in self._results])
        if mapped == "sqrt_s":
            return np.sqrt(np.array([res.s for res in self._results]))
        if mapped == "diagnostics":
            return [res.diagnostics for res in self._results]
        return np.array([getattr(res, mapped) for res in self._results])

    def to_dicts(self) -> list[dict]:
        return [res.to_dict() for res in self._results]
