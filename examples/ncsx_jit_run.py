"""Run a JIT-compiled NEO_JAX solve on the NCSX fixture."""

from neo_jax.examples import ncsx_jit_demo


def main() -> None:
    ncsx_jit_demo(save_path="ncsx_eps_eff.png")


if __name__ == "__main__":
    main()
