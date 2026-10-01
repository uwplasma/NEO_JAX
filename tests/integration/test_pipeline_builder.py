from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neo_jax import NeoConfig, build_vmec_boozer_neo_jax
from neo_jax.pipeline import booz_xform_from_vmec_state_jax, booz_xform_from_vmec_wout


def test_build_vmec_boozer_neo_jax(vmex_equilibrium):
    run = vmex_equilibrium
    config = NeoConfig(surfaces=[np.float64(0.6)], theta_n=8, phi_n=8, npart=8, multra=1,
                       nstep_per=4, nstep_min=20, nstep_max=40, no_bins=10, acc_req=0.1)
    solver = build_vmec_boozer_neo_jax(
        run, booz_kwargs=dict(mboz=4, nboz=0), neo_config=config, jit=True)
    outputs = solver(run.state)
    assert np.all(np.isfinite(outputs.eps_eff))
    assert outputs.diagnostics["rational_surface_policy"] == "error"
    np.testing.assert_allclose(outputs.diagnostics["s"], [0.5625])
    one = booz_xform_from_vmec_state_jax(vmec_run=run, mboz=4, nboz=0,
                                       surfaces=[5], jit=False)
    full = booz_xform_from_vmec_state_jax(vmec_run=run, mboz=4, nboz=0, jit=False)
    for name in ("rmnc_b", "zmns_b", "pmns_b", "bmnc_b"):
        np.testing.assert_allclose(one[name], np.asarray(full[name])[[4]])


def test_pipeline_rejects_asymmetry(vmex_equilibrium):
    run = vmex_equilibrium
    setup = replace(run.runtime.setup, lasym=True)
    asymmetric = replace(run, runtime=replace(run.runtime, setup=setup))
    with pytest.raises(ValueError, match="symmetric"):
        build_vmec_boozer_neo_jax(asymmetric)
    with pytest.raises(ValueError, match="symmetric"):
        booz_xform_from_vmec_wout(replace(run.wout, lasym=True))


def test_compiled_pipeline_preserves_work_limit(vmex_equilibrium):
    solver = build_vmec_boozer_neo_jax(
        vmex_equilibrium, booz_kwargs=dict(mboz=4, nboz=0),
        neo_config=NeoConfig(surfaces=[5], max_rational_field_periods=1))
    with pytest.raises(RuntimeError, match="rational-surface"):
        solver(vmex_equilibrium.state)


def test_compiled_current_pipeline_requires_explicit_work_limit(vmex_equilibrium):
    run = vmex_equilibrium
    setup = replace(run.runtime.setup, ncurr=1)
    current_driven = replace(run, runtime=replace(run.runtime, setup=setup))
    with pytest.raises(ValueError, match="max_rational_field_periods=0"):
        build_vmec_boozer_neo_jax(current_driven)
