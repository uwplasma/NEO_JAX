from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import neo_jax.fixtures as fixtures


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_resolve_external_fixture_prefers_repo_checkout(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    payload = b"fixture bytes from repo"
    repo_path = repo_root / "tests" / "fixtures" / "toy" / "payload.bin"
    repo_path.parent.mkdir(parents=True)
    repo_path.write_bytes(payload)

    spec = fixtures.ExternalFixture(
        name="toy",
        relative_path="tests/fixtures/toy/payload.bin",
        url="https://example.invalid/payload.bin",
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        description="toy fixture",
    )

    monkeypatch.setattr(fixtures, "_FIXTURES", {"toy": spec})
    monkeypatch.setattr(fixtures, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(fixtures, "_cache_root", lambda: tmp_path / "cache")

    assert fixtures.resolve_external_fixture("toy", download=False) == repo_path


def test_resolve_external_fixture_uses_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    payload = b"fixture bytes from cache"
    cache_path = cache_root / "tests" / "fixtures" / "toy" / "payload.bin"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(payload)

    spec = fixtures.ExternalFixture(
        name="toy",
        relative_path="tests/fixtures/toy/payload.bin",
        url="https://example.invalid/payload.bin",
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
        description="toy fixture",
    )

    monkeypatch.setattr(fixtures, "_FIXTURES", {"toy": spec})
    monkeypatch.setattr(fixtures, "_repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(fixtures, "_cache_root", lambda: cache_root)

    assert fixtures.resolve_external_fixture("toy", download=False) == cache_path


def test_resolve_external_fixture_reports_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = fixtures.ExternalFixture(
        name="toy",
        relative_path="tests/fixtures/toy/payload.bin",
        url="https://example.invalid/payload.bin",
        sha256=_sha256_bytes(b"payload"),
        size_bytes=7,
        description="toy fixture",
    )

    monkeypatch.setattr(fixtures, "_FIXTURES", {"toy": spec})
    monkeypatch.setattr(fixtures, "_repo_root", lambda: tmp_path / "repo")
    monkeypatch.setattr(fixtures, "_cache_root", lambda: tmp_path / "cache")

    with pytest.raises(FileNotFoundError, match="NEO_JAX_FETCH_EXTERNAL_FIXTURES=1"):
        fixtures.resolve_external_fixture("toy", download=False)
