"""I/O helpers for NEO_JAX."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Sequence

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
        lasym = "lasym__logical__" in ds.variables and bool(ds.variables["lasym__logical__"][...])
        nfp = int(ds.variables["nfp_b"][:])
        ns_b = int(ds.variables["ns_b"][:])
        mboz_b = int(ds.variables["mboz_b"][:])
        nboz_b = int(ds.variables["nboz_b"][:])

        ixm_b = np.array(ds.variables["ixm_b"][:], dtype=int)
        ixn_b = np.array(ds.variables["ixn_b"][:], dtype=int)

        iota_b = np.array(ds.variables["iota_b"][:], dtype=float)
        buco_b = np.array(ds.variables["buco_b"][:], dtype=float)
        bvco_b = np.array(ds.variables["bvco_b"][:], dtype=float)
        pres_b = np.array(ds.variables["pres_b"][:], dtype=float) if "pres_b" in ds.variables else None

        mode_dims = (ds.variables["ixm_b"].dimensions[0], "mn_mode", "mn_modes")
        def coefficient(name):
            variable = ds.variables[name]
            value = np.asarray(variable[:], dtype=float)
            return value.T if variable.dimensions[0] in mode_dims else value
        rmnc_raw, zmns_raw, pmns_raw, bmnc_raw = (
            coefficient(name) for name in ("rmnc_b", "zmns_b", "pmns_b", "bmnc_b"))
        gmn_raw = coefficient("gmn_b") if "gmn_b" in ds.variables else None
        asymmetric = {name: coefficient(raw) for name, raw in
                      (("rmns", "rmns_b"), ("zmnc", "zmnc_b"),
                       ("lmnc", "pmnc_b"), ("bmns", "bmns_b"))} if lasym else {}

        if "jlist" in ds.variables:
            jlist = np.array(ds.variables["jlist"][:], dtype=int)
        else:
            jlist = np.arange(1, rmnc_raw.shape[0] + 1, dtype=int)

    pack_len = len(jlist)
    rmnc_pack = _transpose_if_needed(rmnc_raw, pack_len)
    zmns_pack = _transpose_if_needed(zmns_raw, pack_len)
    pmns_pack = _transpose_if_needed(pmns_raw, pack_len)
    bmnc_pack = _transpose_if_needed(bmnc_raw, pack_len)
    gmn_pack = _transpose_if_needed(gmn_raw, pack_len) if gmn_raw is not None else None

    max_m = max_m_mode if max_m_mode > 0 else mboz_b - 1
    max_n = max_n_mode if max_n_mode > 0 else nboz_b * nfp
    mode_mask = _select_modes(ixm_b, ixn_b, max_m, max_n)

    ixm = ixm_b[mode_mask]
    ixn = ixn_b[mode_mask]
    mode0 = np.where((ixm_b == 0) & (ixn_b == 0))[0]
    mode0_idx = int(mode0[0]) if len(mode0) else None

    pack_index = {int(surf): idx for idx, surf in enumerate(jlist)}

    if fluxs_arr is not None:
        surfaces = list(fluxs_arr)
    else:
        surfaces = list(jlist)
    packed_rows = [pack_index[int(surf)] for surf in surfaces if int(surf) in pack_index]
    asymmetric = {name: _transpose_if_needed(value, pack_len)[packed_rows][:, mode_mask]
                  for name, value in asymmetric.items()}
    if lasym:
        asymmetric["lmnc"] *= -nfp / (2.0 * math.pi)

    rmnc = []
    zmns = []
    lmns = []
    bmnc = []
    es = []
    iota = []
    curr_pol = []
    curr_tor = []
    pprime = []
    sqrtg00 = []

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
        if pres_b is not None and surf < ns_b:
            pprime.append((pres_b[surf] - pres_b[surf - 1]) / hs)
        else:
            pprime.append(0.0)
        if gmn_pack is not None and mode0_idx is not None:
            sqrtg00.append(float(gmn_pack[pack_idx, mode0_idx]))
        else:
            sqrtg00.append(0.0)

    return BoozerData(
        **asymmetric,
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
        pprime=np.asarray(pprime),
        sqrtg00=np.asarray(sqrtg00),
    )


def _packed_boozer_profiles(booz, ns, xp, *profiles):
    """Align full radial profiles with packed spectra; retain packed profiles."""
    get = booz.get if isinstance(booz, dict) else lambda name, default=None: getattr(booz, name, default)
    def array(value):
        return xp.asarray(value.filled() if isinstance(value, np.ma.MaskedArray) else value)
    jlist, compute_surfs, s_b = get("jlist"), get("compute_surfs"), get("s_b")
    size_field = "ns_b" if jlist is not None else "ns_in"
    if (s_b is None and (jlist is not None or compute_surfs is not None)
            and get(size_field) is None and all(p.shape[0] == ns for p in profiles)):
        raise ValueError("Packed profiles require radial size metadata or s_b")
    if jlist is not None:
        indices = array(jlist).astype(int) - 1
        ns_full = get("ns_b", max(p.shape[0] for p in profiles))
        labels = indices + 1
    elif compute_surfs is not None:
        indices = array(compute_surfs).astype(int)
        ns_full = get("ns_in", max(p.shape[0] for p in profiles)) + 1
        labels = indices + 2
    else:
        if any(p.shape[0] != ns for p in profiles):
            raise ValueError("Full radial profiles require jlist or compute_surfs")
        indices, labels = xp.arange(ns), xp.arange(ns) + 1
        ns_full = get("ns_b", ns)
    if indices.shape != (ns,):
        raise ValueError("Surface metadata must match the packed coefficient rows")
    packed = []
    for profile in profiles:
        if profile.shape[0] != ns or (s_b is None and jlist is not None and profile.shape[0] == int(ns_full)):
            if (not (_JAX_AVAILABLE and isinstance(indices, jax.core.Tracer))
                    and (np.any(np.asarray(indices) < 0) or np.any(np.asarray(indices) >= len(profile)))):
                raise ValueError("Surface metadata is outside the radial profiles")
            profile = xp.take(profile, indices, axis=0)
        packed.append(profile)
    es = array(s_b) if s_b is not None else (labels - 1.5) / max(1, int(ns_full) - 1)
    if es.shape != (ns,):
        raise ValueError("s_b must match the packed coefficient rows")
    return es, *packed



def _convert_boozer(
    booz, *, use_jax, max_m_mode=0, max_n_mode=0, fluxs_arr=None,
    nfp_override=None, mode_indices=None, asym_override=None, mode_first=None,
):
    xp = jnp if use_jax else np

    def get(name, default=None):
        return booz.get(name, default) if isinstance(booz, dict) else getattr(booz, name, default)

    def array(name, dtype=None):
        value = get(name)
        if value is None:
            raise KeyError(f"Missing field {name} in Boozer data")
        return xp.asarray(value.filled() if isinstance(value, np.ma.MaskedArray) else value,
                          dtype=dtype)

    nfp = int(nfp_override if nfp_override is not None else np.asarray(get("nfp_b")))
    ixm, ixn = array("ixm_b", int), array("ixn_b", int)
    if mode_first is None:
        mode_first = bool(get("mode_first", not isinstance(booz, dict)))

    def packed(name):
        value = array(name)
        if value.shape[1] != len(ixm) or (mode_first and value.shape[0] == len(ixm)):
            value = value.T
        if value.shape[1] != len(ixm):
            raise ValueError("Boozer coefficients must have one column per mode")
        return value

    rmnc = packed("rmnc_b")
    ns = rmnc.shape[0]
    es, iota, curr_tor, curr_pol = _packed_boozer_profiles(
        booz, ns, xp, array("iota_b"), array("buco_b"), array("bvco_b"))
    if fluxs_arr and any(not 1 <= int(s) <= ns for s in fluxs_arr):
        raise ValueError("Surface index is outside the Boozer data")
    rows = xp.asarray([int(s)-1 for s in fluxs_arr], dtype=int) if fluxs_arr else xp.arange(ns)
    if mode_indices is None:
        max_m = max_m_mode if max_m_mode > 0 else xp.max(xp.abs(ixm))
        max_n = max_n_mode if max_n_mode > 0 else xp.max(xp.abs(ixn))
        mask = (xp.abs(ixm) <= max_m) & (xp.abs(ixn) <= max_n)
        modes = None
    else:
        modes = xp.asarray(mode_indices, dtype=int)

    def select(value):
        value = xp.take(value, rows, axis=0)
        return value[:, mask] if modes is None else xp.take(value, modes, axis=1)

    asymmetric = (bool(get("asym", get("lasym", get("bmns_b") is not None)))
                  if asym_override is None else bool(asym_override))
    names = {"rmnc": "rmnc_b", "zmns": "zmns_b", "lmns": "pmns_b", "bmnc": "bmnc_b"}
    if asymmetric:
        names.update(rmns="rmns_b", zmnc="zmnc_b", lmnc="pmnc_b", bmns="bmns_b")
    coefficients = {name: select(rmnc if name == "rmnc" else packed(raw))
                    for name, raw in names.items()}
    for name in ("lmns", "lmnc"):
        if name in coefficients:
            coefficients[name] *= -nfp / (2.0 * math.pi)
    return BoozerData(
        **coefficients, nfp=nfp, es=xp.take(es, rows),
        ixm=ixm[mask] if modes is None else xp.take(ixm, modes),
        ixn=ixn[mask] if modes is None else xp.take(ixn, modes),
        iota=xp.take(iota, rows), curr_pol=xp.take(curr_pol, rows),
        curr_tor=xp.take(curr_tor, rows),
    )


def booz_xform_to_boozerdata(
    booz: object, *, max_m_mode: int = 0, max_n_mode: int = 0,
    fluxs_arr: Optional[Sequence[int]] = None, use_jax: bool | None = None,
    mode_first: bool | None = None,
) -> BoozerData:
    """Convert Boozer arrays; square mappings are surface-first, objects mode-first."""
    sample = booz.get("rmnc_b") if isinstance(booz, dict) else getattr(booz, "rmnc_b")
    if use_jax is None:
        use_jax = _JAX_AVAILABLE and isinstance(sample, jax.Array)
    return _convert_boozer(booz, use_jax=use_jax, max_m_mode=max_m_mode,
                           max_n_mode=max_n_mode, fluxs_arr=fluxs_arr, mode_first=mode_first)


def booz_xform_to_boozerdata_jax(
    booz: object, *, max_m_mode: int = 0, max_n_mode: int = 0,
    fluxs_arr: Optional[Sequence[int]] = None, nfp_override: int | None = None,
    mode_indices: Optional[Sequence[int]] = None, asym_override: bool | None = None,
    mode_first: bool | None = None,
) -> BoozerData:
    """Convert traced Boozer outputs with static symmetry and mode selection."""
    if not _JAX_AVAILABLE:  # pragma: no cover - optional
        raise ImportError("JAX is required for booz_xform_to_boozerdata_jax")
    return _convert_boozer(booz, use_jax=True, max_m_mode=max_m_mode,
                           max_n_mode=max_n_mode, fluxs_arr=fluxs_arr,
                           nfp_override=nfp_override, mode_indices=mode_indices,
                           asym_override=asym_override, mode_first=mode_first)
