"""Helpers for optional large test and example fixtures."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


_FETCH_ENV = "NEO_JAX_FETCH_EXTERNAL_FIXTURES"
_CACHE_ENV = "NEO_JAX_FIXTURE_CACHE_DIR"


@dataclass(frozen=True)
class ExternalFixture:
    name: str
    relative_path: str
    url: str
    sha256: str
    size_bytes: int
    description: str


_FIXTURES: dict[str, ExternalFixture] = {
    "ncsx_boozmn": ExternalFixture(
        name="ncsx_boozmn",
        relative_path="tests/fixtures/ncsx/boozmn_ncsx_c09r00_free.nc",
        url=(
            "https://github.com/uwplasma/NEO_JAX/releases/download/"
            "large-fixtures-v1/boozmn_ncsx_c09r00_free.nc"
        ),
        sha256="61d7fe4981811317d9f3b191f7ab497665f103a7c34834c71592fcc51abddb8b",
        size_bytes=13_771_012,
        description="NCSX Boozer tutorial/reference equilibrium",
    )
}


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _cache_root() -> Path:
    configured = os.getenv(_CACHE_ENV)
    if configured:
        return Path(configured).expanduser().resolve()
    return Path.home() / ".cache" / "neo_jax" / "fixtures"


def _cache_path(spec: ExternalFixture) -> Path:
    return _cache_root() / spec.relative_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, spec: ExternalFixture) -> None:
    if _sha256(path) != spec.sha256:
        raise ValueError(
            f"Fixture checksum mismatch for {path}. "
            f"Expected {spec.sha256} for {spec.name}."
        )


def _download(spec: ExternalFixture, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(delete=False, dir=destination.parent, suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        with urlopen(spec.url, timeout=120) as response, tmp_path.open("wb") as out:
            shutil.copyfileobj(response, out)
        _verify(tmp_path, spec)
        tmp_path.replace(destination)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return destination


def resolve_external_fixture(name: str, *, download: bool | None = None) -> Path:
    """Return a path to an optional large fixture.

    If the fixture is still present in the repository checkout, that path is
    returned directly. Otherwise the helper looks in the local fixture cache
    and, if allowed, downloads the file from the NEO_JAX release assets.
    """

    try:
        spec = _FIXTURES[name]
    except KeyError as exc:
        raise KeyError(f"Unknown external fixture: {name}") from exc

    repo_path = _repo_root() / spec.relative_path
    if repo_path.exists():
        return repo_path

    cached = _cache_path(spec)
    if cached.exists():
        _verify(cached, spec)
        return cached

    allow_download = _is_truthy(os.getenv(_FETCH_ENV)) if download is None else download
    if not allow_download:
        raise FileNotFoundError(
            f"External fixture '{name}' is not present in the checkout. "
            f"Enable download with {_FETCH_ENV}=1 or call "
            f"resolve_external_fixture({name!r}, download=True). "
            f"The file will be cached under {cached}."
        )

    return _download(spec, cached)


def ncsx_boozmn_path(*, download: bool | None = None) -> Path:
    """Return the NCSX Boozer tutorial/reference equilibrium path."""

    return resolve_external_fixture("ncsx_boozmn", download=download)
