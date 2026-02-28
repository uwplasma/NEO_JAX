from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp


def test_vmec_boozer_neo_jax_grad():
    pytest.importorskip("vmec_jax")
    pytest.importorskip("booz_xform_jax")

    from vmec_jax.driver import load_example

    example = load_example("circular_tokamak", with_wout=True)
    if example.state is None or example.wout is None:
        pytest.skip("No reference VMEC state available")

    from vmec_jax.booz_input import booz_xform_inputs_from_state
    from booz_xform_jax.jax_api import prepare_booz_xform_constants, booz_xform_jax_impl
    from neo_jax.config import NeoConfig
    from neo_jax.driver import run_neo_from_boozer_jax
    from neo_jax.io import booz_xform_to_boozerdata

    signgs = int(getattr(example.wout, "signgs", 1))
    state0 = example.state
    static = example.static
    indata = example.indata

    base_inputs = booz_xform_inputs_from_state(
        state=state0,
        static=static,
        indata=indata,
        signgs=signgs,
    )

    ns_b = int(np.asarray(base_inputs.rmnc).shape[0])
    if ns_b < 2:
        pytest.skip("No half-mesh surfaces available")
    surf_idx = int(max(0, min(ns_b - 1, ns_b // 2)))
    idx0 = max(0, surf_idx - 1)
    idx1 = surf_idx
    surface_indices = jnp.asarray([idx0, idx1], dtype=jnp.int32)

    constants, grids = prepare_booz_xform_constants(
        nfp=int(base_inputs.nfp),
        mboz=4,
        nboz=4,
        asym=bool(static.cfg.lasym),
        xm=np.asarray(base_inputs.xm),
        xn=np.asarray(base_inputs.xn),
        xm_nyq=np.asarray(base_inputs.xm_nyq),
        xn_nyq=np.asarray(base_inputs.xn_nyq),
    )

    config = NeoConfig(
        surfaces=[idx0 + 1, idx1 + 1],
        theta_n=8,
        phi_n=8,
        npart=8,
        multra=1,
        nstep_per=4,
        nstep_min=20,
        nstep_max=40,
        no_bins=10,
        acc_req=0.1,
    )
    control = config.to_control()

    def objective(scale):
        factor = 1.0 + 1.0e-2 * scale
        state = state0.__class__(
            layout=state0.layout,
            Rcos=jnp.asarray(state0.Rcos) * factor,
            Rsin=jnp.asarray(state0.Rsin),
            Zcos=jnp.asarray(state0.Zcos),
            Zsin=jnp.asarray(state0.Zsin),
            Lcos=jnp.asarray(state0.Lcos),
            Lsin=jnp.asarray(state0.Lsin),
        )

        inputs = booz_xform_inputs_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs,
        )

        booz_out = booz_xform_jax_impl(
            rmnc=inputs.rmnc,
            zmns=inputs.zmns,
            lmns=inputs.lmns,
            bmnc=inputs.bmnc,
            bsubumnc=inputs.bsubumnc,
            bsubvmnc=inputs.bsubvmnc,
            iota=inputs.iota,
            xm=inputs.xm,
            xn=inputs.xn,
            xm_nyq=inputs.xm_nyq,
            xn_nyq=inputs.xn_nyq,
            constants=constants,
            grids=grids,
            bmns=inputs.bmns,
            bsubumns=inputs.bsubumns,
            bsubvmns=inputs.bsubvmns,
            surface_indices=surface_indices,
        )

        booz = booz_xform_to_boozerdata(booz_out, use_jax=True)
        outputs = run_neo_from_boozer_jax(booz, control)
        return jnp.sum(outputs.eps_eff)

    _, tangent = jax.jvp(objective, (jnp.array(0.0),), (jnp.array(1.0),))
    assert np.isfinite(np.asarray(tangent))
