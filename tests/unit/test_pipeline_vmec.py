from __future__ import annotations

import numpy as np
import pytest

from neo_jax import NeoConfig
from neo_jax.pipeline import booz_xform_from_vmec_wout, run_vmec_boozer_neo


def test_vmec_boozer_pipeline_smoke(vmex_equilibrium, tmp_path):
    from vmex import write_wout

    wout = vmex_equilibrium.wout
    path = tmp_path / "wout.nc"
    write_wout(path, wout)
    kwargs = dict(mboz=4, nboz=0, jit=False)
    config = NeoConfig(theta_n=8, phi_n=8, surfaces=[0.6], npart=8, multra=1,
                       nstep_per=4, nstep_min=20, nstep_max=40, no_bins=10, acc_req=0.1)
    results = [run_vmec_boozer_neo(source, booz_kwargs=kwargs, neo_config=config,
                                 progress=False) for source in (wout, path)]
    np.testing.assert_allclose(results[0].epsilon_effective, results[1].epsilon_effective)
    assert np.all(np.isfinite(results[0].epsilon_effective))
    full = booz_xform_from_vmec_wout(wout, **kwargs)
    selected = booz_xform_from_vmec_wout(wout, surfaces=[1, 5, 8], **kwargs)
    np.testing.assert_allclose(selected["s_b"], np.asarray(full["s_b"])[[0, 4, 7]])
    np.testing.assert_allclose(selected["rmnc_b"], np.asarray(full["rmnc_b"])[[0, 4, 7]])


@pytest.mark.parametrize("surfaces", [[], [0], [9], [-0.1], [1.1], [np.nan]])
def test_invalid_pipeline_surfaces(vmex_equilibrium, surfaces):
    with pytest.raises(ValueError):
        booz_xform_from_vmec_wout(vmex_equilibrium.wout, surfaces=surfaces)


def test_compiled_current_driven_builder_keeps_live_guard(monkeypatch):
    from types import SimpleNamespace

    import jax.numpy as jnp

    from neo_jax import driver, io, pipeline

    run = SimpleNamespace(runtime=SimpleNamespace(setup=SimpleNamespace(ncurr=1, lasym=False)),
                          state=jnp.array(.43))
    grids = SimpleNamespace(xm_b=np.array([0]), xn_b=np.array([0]))
    monkeypatch.setattr(pipeline, "_build_boozer_transform",
                        lambda *args, **kwargs: (lambda state: state, grids, 2))
    monkeypatch.setattr(io, "booz_xform_to_boozerdata_jax", lambda data, **kwargs: data)
    observed = []
    def solve(data, control, **kwargs):
        observed.append(kwargs["max_rational_field_periods"])
        assert kwargs["_work_guard"] is None
        return 2*data
    monkeypatch.setattr(driver, "run_neo_from_boozer_jax", solve)
    evaluate = pipeline.build_vmec_boozer_neo_jax(run, neo_config=NeoConfig(reverse_mode=True))
    np.testing.assert_allclose(evaluate(run.state), .86)
    assert observed == [100_000]
