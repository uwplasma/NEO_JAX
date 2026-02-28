from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


def test_build_vmec_boozer_neo_jax():
    pytest.importorskip("vmec_jax")
    pytest.importorskip("booz_xform_jax")

    from vmec_jax.driver import load_example
    from neo_jax import NeoConfig, build_vmec_boozer_neo_jax

    example = load_example("circular_tokamak", with_wout=True)
    if example.state is None or example.wout is None:
        pytest.skip("No reference VMEC state available")

    signgs = int(getattr(example.wout, "signgs", 1))
    run = SimpleNamespace(
        state=example.state,
        static=example.static,
        indata=example.indata,
        signgs=signgs,
    )

    config = NeoConfig(surfaces=[0.6], theta_n=8, phi_n=8, npart=8, multra=1)
    solver = build_vmec_boozer_neo_jax(
        run,
        booz_kwargs=dict(mboz=4, nboz=4),
        neo_config=config,
        jit=False,
    )
    outputs = solver(run.state)
    eps = np.asarray(outputs.eps_eff)
    assert np.all(np.isfinite(eps))
