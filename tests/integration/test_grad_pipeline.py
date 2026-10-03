from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from neo_jax import NeoConfig, build_vmec_boozer_neo_jax


@pytest.mark.parametrize("vmex_equilibrium", [1], indirect=True)
@pytest.mark.parametrize("sequential", [None, True])
def test_vmec_boozer_neo_jax_grad(vmex_equilibrium, sequential):
    run = vmex_equilibrium
    config = NeoConfig(sequential=sequential, surfaces=[0.6], theta_n=8, phi_n=8, npart=8, multra=1,
                       nstep_per=4, nstep_min=20, nstep_max=40, no_bins=10, acc_req=0.1)
    solver = build_vmec_boozer_neo_jax(
        run, booz_kwargs=dict(mboz=4, nboz=2), neo_config=config, jit=True)

    def objective(scale):
        state = replace(run.state, R_cos=run.state.R_cos * (1 + 0.01 * scale))
        outputs = solver(state)
        return jnp.array([jnp.sum(outputs.eps_eff), jnp.sum(outputs.diagnostics["b_ref"])])

    value, tangent = jax.jvp(objective, (jnp.array(0.0),), (jnp.array(1.0),))
    step = 1e-3
    fd = (objective(step) - objective(-step)) / (2 * step)
    assert np.all(np.isfinite(value)) and np.all(np.isfinite(tangent))
    assert abs(float(tangent[0])) > 1e-10
    assert abs(float(tangent[1])) > 1e-5
    np.testing.assert_allclose(tangent, fd, rtol=1e-5, atol=1e-9)
