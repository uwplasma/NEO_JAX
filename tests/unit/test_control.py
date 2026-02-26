from pathlib import Path

from neo_jax.control import read_control


def test_read_control_orbits():
    path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "orbits" / "neo_in.ORBITS"
    params = read_control(path)

    assert params.theta_n == 50
    assert params.phi_n == 50
    assert params.npart == 40
    assert params.multra == 2
    assert params.nstep_per == 20
    assert params.nstep_min == 200
    assert params.nstep_max == 500
    assert params.fluxs_arr == [2, 4, 8, 16, 32, 64, 96, 120]


def test_read_control_ncsx():
    path = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "ncsx" / "neo_in.ncsx_c09r00_free"
    params = read_control(path)

    assert params.theta_n == 200
    assert params.phi_n == 200
    assert params.npart == 75
    assert params.multra == 1
    assert params.nstep_per == 75
    assert params.nstep_min == 500
    assert params.nstep_max == 2000
