from __future__ import annotations

import numpy as np
import pytest

from neo_jax import NeoConfig
from neo_jax.pipeline import run_vmec_boozer_neo


def test_vmec_boozer_pipeline_smoke():
    pytest.importorskip("vmec_jax")
    pytest.importorskip("booz_xform_jax")

    from vmec_jax.driver import example_paths
    from vmec_jax.wout import read_wout

    _, wout_path = example_paths("circular_tokamak")
    if wout_path is None:
        pytest.skip("No reference wout file found for circular_tokamak")

    wout = read_wout(wout_path)

    config = NeoConfig(theta_n=16, phi_n=16, surfaces=[0.6], write_progress=False)
    booz_kwargs = dict(mboz=4, nboz=4, jit=False)

    results = run_vmec_boozer_neo(
        wout,
        booz_kwargs=booz_kwargs,
        neo_config=config,
        progress=False,
        fast_bcovar=True,
    )

    eps = np.asarray(results.epsilon_effective)
    assert np.all(np.isfinite(eps))
