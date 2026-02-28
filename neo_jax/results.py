"""User-friendly result containers for NEO_JAX."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Sequence

from .data_models import NeoOutputs

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


def _as_array(value):
    try:
        arr = np.asarray(value)
    except Exception:
        return None
    if arr.shape == ():
        return None
    return arr


def neo_outputs_to_results(
    outputs: NeoOutputs,
    *,
    flux_indices: Sequence[int] | None = None,
) -> NeoResults:
    """Convert JAX-friendly ``NeoOutputs`` into ``NeoResults``."""
    diag = outputs.diagnostics or {}

    eps_eff = np.asarray(outputs.eps_eff)
    eps_par = np.asarray(outputs.eps_par)
    ctr_one = np.asarray(outputs.ctr_one)
    ctr_tot = np.asarray(outputs.ctr_tot)

    n = int(eps_eff.shape[0])

    def _series(key: str, default: float = 0.0) -> np.ndarray:
        value = diag.get(key, default)
        arr = np.asarray(value)
        if arr.shape == ():
            return np.full((n,), float(arr))
        return arr

    s = _series("s", 0.0)
    r_eff = _series("r_eff", 0.0)
    iota = _series("iota", 0.0)
    b_ref = _series("b_ref", 0.0)
    r_ref = _series("r_ref", 0.0)
    bareph = _series("bareph", 0.0)
    barept = _series("barept", 0.0)
    yps = _series("yps", 0.0)

    flux_index = None
    if flux_indices is not None:
        flux_index = np.asarray(flux_indices)
    else:
        flux_index = diag.get("flux_index")
        if flux_index is not None:
            flux_index = np.asarray(flux_index)
    if flux_index is None or flux_index.shape == ():
        flux_index = np.arange(1, n + 1, dtype=int)

    results: list[NeoSurfaceResult] = []
    for idx in range(n):
        surface_diag: dict[str, object] = {}
        for key, value in diag.items():
            arr = _as_array(value)
            if arr is not None and arr.shape[0] == n:
                if arr.ndim == 1:
                    surface_diag[key] = float(arr[idx])
                else:
                    surface_diag[key] = arr[idx]
            else:
                surface_diag[key] = value

        results.append(
            NeoSurfaceResult(
                flux_index=int(flux_index[idx]),
                s=float(s[idx]),
                r_eff=float(r_eff[idx]),
                iota=float(iota[idx]),
                b_ref=float(b_ref[idx]),
                r_ref=float(r_ref[idx]),
                epsilon_effective=float(eps_eff[idx]),
                epsilon_effective_by_class=np.asarray(eps_par[idx]),
                ctrone=float(ctr_one[idx]),
                ctrtot=float(ctr_tot[idx]),
                bareph=float(bareph[idx]),
                barept=float(barept[idx]),
                yps=float(yps[idx]),
                diagnostics=surface_diag,
            )
        )

    return NeoResults(results)
