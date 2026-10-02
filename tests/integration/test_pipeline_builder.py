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


@pytest.mark.parametrize("vmex_equilibrium", [(1, True)], indirect=True)
def test_pipeline_preserves_asymmetry(vmex_equilibrium, record_property):
    run = vmex_equilibrium
    kwargs = dict(mboz=4, nboz=2, surfaces=[0.6], jit=True)
    state = booz_xform_from_vmec_state_jax(vmec_run=run, **kwargs)
    wout = booz_xform_from_vmec_wout(run.wout, **kwargs)
    assert state["asym"] and wout["asym"]
    assert np.linalg.norm(state["bmns_b"]) > 1e-8
    errors, absolute_errors = {}, {}
    for cosine, sine in (("rmnc_b", "rmns_b"), ("zmns_b", "zmnc_b"),
                          ("pmns_b", "pmnc_b"), ("bmnc_b", "bmns_b")):
        actual, expected = (np.stack([result[cosine], result[sine]]) for result in (state, wout))
        absolute_errors[cosine] = np.linalg.norm(actual-expected)
        errors[cosine] = absolute_errors[cosine] / np.linalg.norm(expected)
        for name in (cosine, sine):
            record_property(name + "_norm", float(np.linalg.norm(wout[name])))
        record_property(cosine + "_relative_l2", float(errors[cosine]))
        record_property(cosine + "_absolute_l2", float(absolute_errors[cosine]))
    theta = 2*np.pi*np.arange(17)/17
    phi = 2*np.pi*np.arange(13)/(13*run.inp.nfp)
    phase = theta[:, None, None]*np.asarray(state["ixm_b"])
    phase = phase - phi[None, :, None]*np.asarray(state["ixn_b"])
    fields = [np.sum(np.asarray(result["bmnc_b"])[0]*np.cos(phase)
                     + np.asarray(result["bmns_b"])[0]*np.sin(phase), axis=-1)
              for result in (state, wout)]
    field_error = np.linalg.norm(fields[0]-fields[1])/np.linalg.norm(fields[1])
    record_property("B_grid_relative_l2", float(field_error))
    assert field_error < 1e-9
    # The state and WOUT covariant tables use different radial finite differences.
    assert max(absolute_errors.values()) < 1e-7, absolute_errors


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
