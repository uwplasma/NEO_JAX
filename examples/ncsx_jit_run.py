"""Run a JIT-compiled NEO_JAX solve on the NCSX fixture."""

from dataclasses import replace
from pathlib import Path

from neo_jax.control import read_control
from neo_jax.driver import run_neo_from_boozmn


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    control_path = repo_root / "tests" / "fixtures" / "ncsx" / "neo_in.ncsx_c09r00_free"
    boozmn_path = repo_root / "tests" / "fixtures" / "ncsx" / "boozmn_ncsx_c09r00_free.nc"

    control = read_control(control_path)
    control = replace(
        control,
        fluxs_arr=[9],
        write_progress=1,
        write_integrate=0,
        write_diagnostic=0,
        write_output_files=0,
    )

    results = run_neo_from_boozmn(
        str(boozmn_path),
        control,
        use_jax=True,
        progress=True,
    )

    result = results[0]
    print("NEO_JAX JIT epstot:", result["epstot"])


if __name__ == "__main__":
    main()
