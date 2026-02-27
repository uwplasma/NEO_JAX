"""I/O helpers for NEO_JAX."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import math
import numpy as np

try:  # Optional JAX support for end-to-end pipelines
    import jax
    import jax.numpy as jnp

    _JAX_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    jax = None  # type: ignore
    jnp = None  # type: ignore
    _JAX_AVAILABLE = False

from .data_models import BoozerData

try:
    import netCDF4  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    netCDF4 = None


def _require_netcdf4() -> None:
    if netCDF4 is None:
        raise ImportError("netCDF4 is required to read boozmn files")


def resolve_control_path(extension: Optional[str] = None) -> Path:
    """Resolve NEO control file paths following xneo conventions."""
    if not extension:
        path = Path("neo.in")
        if path.exists():
            return path
        raise FileNotFoundError("neo.in not found")

    candidates = [
        Path(f"neo_param.{extension}"),
        Path("neo_param.in"),
        Path(f"neo_in.{extension}"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No control file found for extension '{extension}'")


def _extension_candidates(base: str | Path, extension: str) -> list[Path]:
    base_str = str(base)
    candidates: list[str] = []

    if base_str in extension:
        candidates.append(extension)
    else:
        if extension.startswith((".", "_")):
            candidates.append(f"{base_str}{extension}")
        else:
            candidates.append(f"{base_str}_{extension}")
        candidates.append(f"{base_str}.{extension}")

    expanded: list[Path] = []
    for cand in candidates:
        path = Path(cand)
        expanded.append(path)
        if path.suffix != ".nc":
            expanded.append(path.with_suffix(".nc"))
    return expanded


def resolve_boozmn_path(base: str | Path, extension: str | None = None) -> Path:
    """Resolve a boozmn file from a base name and optional extension."""
    path = Path(base)
    if path.exists():
        return path

    if extension:
        for candidate in _extension_candidates(path, extension):
            if candidate.exists():
                return candidate

    if path.suffix != ".nc":
        nc_path = path.with_suffix(".nc")
        if nc_path.exists():
            return nc_path

    if path.name != "boozmn":
        for candidate in (Path("boozmn"), Path("boozmn.nc")):
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Boozmn file not found: {base}")


def read_boozmn_metadata(path: str | Path) -> dict:
    """Read minimal metadata (ns_b, jlist) from a boozmn file."""
    _require_netcdf4()
    booz_path = resolve_boozmn_path(path, None)
    with netCDF4.Dataset(booz_path) as ds:  # type: ignore[union-attr]
        ns_b = int(ds.variables["ns_b"][:])
        if "jlist" in ds.variables:
            jlist = np.array(ds.variables["jlist"][:], dtype=int).tolist()
        else:
            jlist = list(range(1, ns_b + 1))
    return {"ns_b": ns_b, "jlist": jlist}


def _transpose_if_needed(arr: np.ndarray, pack_len: int) -> np.ndarray:
    if arr.shape[0] == pack_len:
        return arr
    if arr.shape[1] == pack_len:
        return arr.T
    raise ValueError("Unexpected boozmn array shape")


def _select_modes(ixm: np.ndarray, ixn: np.ndarray, max_m: int, max_n: int):
    if _JAX_AVAILABLE and isinstance(ixm, jax.Array):  # type: ignore[arg-type]
        return (jnp.abs(ixm) <= max_m) & (jnp.abs(ixn) <= max_n)  # type: ignore[union-attr]
    return (np.abs(ixm) <= max_m) & (np.abs(ixn) <= max_n)


def read_boozmn(
    path: str | Path,
    *,
    max_m_mode: int = 0,
    max_n_mode: int = 0,
    fluxs_arr: Optional[Sequence[int]] = None,
    extension: str | None = None,
) -> BoozerData:
    """Read a boozmn netCDF file and return BoozerData.

    This function follows the packing conventions used in STELLOPT's
    read_boozer_mod, but only retains the surfaces requested.
    """
    _require_netcdf4()
    booz_path = resolve_boozmn_path(path, extension)

    with netCDF4.Dataset(booz_path) as ds:  # type: ignore[union-attr]
        nfp = int(ds.variables["nfp_b"][:])
        ns_b = int(ds.variables["ns_b"][:])
        mboz_b = int(ds.variables["mboz_b"][:])
        nboz_b = int(ds.variables["nboz_b"][:])

        ixm_b = np.array(ds.variables["ixm_b"][:], dtype=int)
        ixn_b = np.array(ds.variables["ixn_b"][:], dtype=int)

        iota_b = np.array(ds.variables["iota_b"][:], dtype=float)
        buco_b = np.array(ds.variables["buco_b"][:], dtype=float)
        bvco_b = np.array(ds.variables["bvco_b"][:], dtype=float)

        rmnc_raw = np.array(ds.variables["rmnc_b"][:], dtype=float)
        zmns_raw = np.array(ds.variables["zmns_b"][:], dtype=float)
        pmns_raw = np.array(ds.variables["pmns_b"][:], dtype=float)
        bmnc_raw = np.array(ds.variables["bmnc_b"][:], dtype=float)

        if "jlist" in ds.variables:
            jlist = np.array(ds.variables["jlist"][:], dtype=int)
        else:
            jlist = np.arange(1, rmnc_raw.shape[0] + 1, dtype=int)

    pack_len = rmnc_raw.shape[0]
    rmnc_pack = _transpose_if_needed(rmnc_raw, pack_len)
    zmns_pack = _transpose_if_needed(zmns_raw, pack_len)
    pmns_pack = _transpose_if_needed(pmns_raw, pack_len)
    bmnc_pack = _transpose_if_needed(bmnc_raw, pack_len)

    max_m = max_m_mode if max_m_mode > 0 else mboz_b - 1
    max_n = max_n_mode if max_n_mode > 0 else nboz_b * nfp
    mode_mask = _select_modes(ixm_b, ixn_b, max_m, max_n)

    ixm = ixm_b[mode_mask]
    ixn = ixn_b[mode_mask]

    pack_index = {int(surf): idx for idx, surf in enumerate(jlist)}

    if fluxs_arr is not None:
        surfaces = list(fluxs_arr)
    else:
        surfaces = list(jlist)

    rmnc = []
    zmns = []
    lmns = []
    bmnc = []
    es = []
    iota = []
    curr_pol = []
    curr_tor = []

    hs = 1.0 / (ns_b - 1)
    for surf in surfaces:
        pack_idx = pack_index.get(int(surf))
        if pack_idx is None:
            raise ValueError(f"Surface {surf} not found in boozmn jlist")

        rmnc.append(rmnc_pack[pack_idx, mode_mask])
        zmns.append(zmns_pack[pack_idx, mode_mask])
        lmns.append(-pmns_pack[pack_idx, mode_mask] * nfp / (2.0 * math.pi))
        bmnc.append(bmnc_pack[pack_idx, mode_mask])

        es.append((surf - 1.5) * hs)
        iota.append(iota_b[surf - 1])
        curr_pol.append(bvco_b[surf - 1])
        curr_tor.append(buco_b[surf - 1])

    return BoozerData(
        rmnc=np.asarray(rmnc),
        zmns=np.asarray(zmns),
        lmns=np.asarray(lmns),
        bmnc=np.asarray(bmnc),
        ixm=np.asarray(ixm),
        ixn=np.asarray(ixn),
        es=np.asarray(es),
        iota=np.asarray(iota),
        curr_pol=np.asarray(curr_pol),
        curr_tor=np.asarray(curr_tor),
        nfp=nfp,
    )


def booz_xform_to_boozerdata(
    booz: object,
    *,
    max_m_mode: int = 0,
    max_n_mode: int = 0,
    fluxs_arr: Optional[Sequence[int]] = None,
    use_jax: bool | None = None,
) -> BoozerData:
    """Convert booz_xform-style arrays into BoozerData.

    The input object can be a mapping or an object with attributes matching the
    boozmn variable names (e.g., ``rmnc_b``, ``zmns_b``, ``pmns_b``).
    """

    def _get(name: str):
        if isinstance(booz, dict) and name in booz:
            return booz[name]
        if hasattr(booz, name):
            return getattr(booz, name)
        raise KeyError(f"Missing field {name} in Boozer data")

    def _asarray(obj, *, dtype=None):
        if isinstance(obj, np.ma.MaskedArray):
            obj = obj.filled()
        return xp.asarray(obj, dtype=dtype)

    sample = _get("rmnc_b")
    if use_jax is None:
        use_jax = _JAX_AVAILABLE and isinstance(sample, jax.Array)  # type: ignore[arg-type]

    xp = jnp if (use_jax and _JAX_AVAILABLE) else np

    nfp = int(np.asarray(_get("nfp_b")).squeeze())
    ixm_b = _asarray(_get("ixm_b"), dtype=int)
    ixn_b = _asarray(_get("ixn_b"), dtype=int)

    iota_b = _asarray(_get("iota_b"), dtype=float)
    buco_b = _asarray(_get("buco_b"), dtype=float)
    bvco_b = _asarray(_get("bvco_b"), dtype=float)

    rmnc_raw = _asarray(_get("rmnc_b"), dtype=float)
    zmns_raw = _asarray(_get("zmns_b"), dtype=float)
    pmns_raw = _asarray(_get("pmns_b"), dtype=float)
    bmnc_raw = _asarray(_get("bmnc_b"), dtype=float)

    if rmnc_raw.shape[0] == ixm_b.shape[0]:
        rmnc_raw = rmnc_raw.T
        zmns_raw = zmns_raw.T
        pmns_raw = pmns_raw.T
        bmnc_raw = bmnc_raw.T

    ns_b = rmnc_raw.shape[0]

    max_m = max_m_mode if max_m_mode > 0 else int(np.max(np.abs(np.asarray(ixm_b))))
    max_n = max_n_mode if max_n_mode > 0 else int(np.max(np.abs(np.asarray(ixn_b))))
    mode_mask = _select_modes(ixm_b, ixn_b, max_m, max_n)

    ixm = ixm_b[mode_mask]
    ixn = ixn_b[mode_mask]

    if fluxs_arr:
        surfaces = list(fluxs_arr)
    else:
        surfaces = list(range(1, ns_b + 1))

    rmnc = []
    zmns = []
    lmns = []
    bmnc = []
    es = []
    iota = []
    curr_pol = []
    curr_tor = []

    hs = 1.0 / (ns_b - 1)
    for surf in surfaces:
        surf_idx = surf - 1
        rmnc.append(rmnc_raw[surf_idx, mode_mask])
        zmns.append(zmns_raw[surf_idx, mode_mask])
        lmns.append(-pmns_raw[surf_idx, mode_mask] * nfp / (2.0 * math.pi))
        bmnc.append(bmnc_raw[surf_idx, mode_mask])

        es.append((surf - 1.5) * hs)
        iota.append(iota_b[surf_idx])
        curr_pol.append(bvco_b[surf_idx])
        curr_tor.append(buco_b[surf_idx])

    arr = xp.asarray

    return BoozerData(
        rmnc=arr(rmnc),
        zmns=arr(zmns),
        lmns=arr(lmns),
        bmnc=arr(bmnc),
        ixm=arr(ixm),
        ixn=arr(ixn),
        es=arr(es),
        iota=arr(iota),
        curr_pol=arr(curr_pol),
        curr_tor=arr(curr_tor),
        nfp=nfp,
    )
