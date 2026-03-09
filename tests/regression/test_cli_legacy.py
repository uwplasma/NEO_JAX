"""Legacy CLI parity tests against the STELLOPT ``xneo`` executable."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[2]
REFERENCE_BIN = Path(os.environ.get("NEO_REFERENCE_BIN", str(Path.home() / "bin" / "xneo")))


def _run_neo_pair(tmp_path: Path, *, extension: str) -> tuple[Path, Path]:
    ref_dir = tmp_path / "ref"
    jax_dir = tmp_path / "jax"
    ref_dir.mkdir(exist_ok=True)
    jax_dir.mkdir(exist_ok=True)

    subprocess.run(
        [str(REFERENCE_BIN), extension],
        cwd=ref_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    subprocess.run(
        ["python", "-m", "neo_jax", extension],
        cwd=jax_dir,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return ref_dir, jax_dir


def _load_numeric_file(path: Path) -> np.ndarray:
    return np.fromstring(path.read_text().replace("D", "E"), sep=" ")


def _assert_exact_text(ref_dir: Path, jax_dir: Path, names: list[str]) -> None:
    for name in names:
        assert (ref_dir / name).read_text() == (jax_dir / name).read_text(), name


def _assert_numeric_close(ref_dir: Path, jax_dir: Path, names: list[str], *, atol: float = 1e-12) -> None:
    for name in names:
        ref = _load_numeric_file(ref_dir / name)
        got = _load_numeric_file(jax_dir / name)
        assert ref.shape == got.shape, name
        assert np.allclose(ref, got, rtol=0.0, atol=atol), name


@pytest.mark.skipif(not REFERENCE_BIN.exists(), reason=f"Reference xneo binary not found at {REFERENCE_BIN}")
def test_cli_landreman_matches_xneo(tmp_path: Path) -> None:
    fixture = REPO / "tests" / "fixtures" / "landreman_qa_lowres"
    extension = "LandremanPaul2021_QA_lowres"

    for td in (tmp_path / "ref", tmp_path / "jax"):
        td.mkdir()
        shutil.copy2(fixture / f"neo_in.{extension}", td / f"neo_in.{extension}")
        shutil.copy2(fixture / f"boozmn_{extension}.nc", td / f"boozmn_{extension}.nc")

    ref_dir, jax_dir = _run_neo_pair(tmp_path, extension=extension)
    assert sorted(path.name for path in ref_dir.iterdir()) == sorted(path.name for path in jax_dir.iterdir())
    _assert_exact_text(
        ref_dir,
        jax_dir,
        [f"neo_out.{extension}", f"neolog.{extension}"],
    )


@pytest.mark.skipif(not REFERENCE_BIN.exists(), reason=f"Reference xneo binary not found at {REFERENCE_BIN}")
def test_cli_mini_orbits_matches_xneo_outputs(tmp_path: Path) -> None:
    extension = "ORBITS_MINI"
    booz_src = REPO / "tests" / "fixtures" / "orbits" / "boozmn_ORBITS_FAST.nc"
    control_text = (
        textwrap.dedent(
            """
            ! mini legacy parity case
            ! line2
            ! line3
            boozmn
            neo_out.ORBITS_MINI
            1
            96
            8
            8
            0
            0
            10
            2
            0.05
            10
            5
            10
            20
            0
            1
            0
            0
            2
            0
            1
            0
            1
            1
            0
            0
            0
            0
            neo_cur.ORBITS_MINI
            200
            2
            0
            """
        ).strip()
        + "\n"
    )

    for td in (tmp_path / "ref", tmp_path / "jax"):
        td.mkdir()
        (td / f"neo_in.{extension}").write_text(control_text, encoding="utf-8")
        shutil.copy2(booz_src, td / f"boozmn_{extension}.nc")

    ref_dir, jax_dir = _run_neo_pair(tmp_path, extension=extension)

    assert sorted(path.name for path in ref_dir.iterdir()) == sorted(path.name for path in jax_dir.iterdir())
    _assert_exact_text(
        ref_dir,
        jax_dir,
        [
            f"neo_out.{extension}",
            "diagnostic.dat",
            "diagnostic_add.dat",
            "diagnostic_bigint.dat",
            "conver.dat",
            f"neolog.{extension}",
        ],
    )
    _assert_numeric_close(
        ref_dir,
        jax_dir,
        [
            "dimension.dat",
            "es_arr.dat",
            "mn_arr.dat",
            "rmnc_arr.dat",
            "zmns_arr.dat",
            "lmns_arr.dat",
            "bmnc_arr.dat",
            "theta_arr.dat",
            "phi_arr.dat",
            "b_s_arr.dat",
            "r_s_arr.dat",
            "z_s_arr.dat",
            "l_s_arr.dat",
            "isqrg_arr.dat",
            "sqrg11_arr.dat",
            "kg_arr.dat",
            "pard_arr.dat",
            "r_tb_arr.dat",
            "z_tb_arr.dat",
            "p_tb_arr.dat",
            "b_tb_arr.dat",
            "r_pb_arr.dat",
            "z_pb_arr.dat",
            "p_pb_arr.dat",
            "b_pb_arr.dat",
            "gtbtb_arr.dat",
            "gpbpb_arr.dat",
            "gtbpb_arr.dat",
        ],
        atol=5e-13,
    )
