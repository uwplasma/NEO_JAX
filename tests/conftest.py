import os
import sys

import jax

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Match Fortran double precision behavior in tests.
jax.config.update("jax_enable_x64", True)


import numpy as np
import pytest


@pytest.fixture(scope="session")
def vmex_equilibrium(request):
    vmex = pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax")
    from vmex import optimize

    case = getattr(request, "param", 0)
    ntor, lasym = case if isinstance(case, tuple) else (int(case), False)
    rbc, zbs = np.zeros((2 * ntor + 1, 4)), np.zeros((2 * ntor + 1, 4))
    rbc[ntor, :2], zbs[ntor, 1] = [6, 1], 1
    if ntor:
        rbc[ntor + 1, 1], zbs[ntor + 1, 1] = 0.1, -0.1
    rbs, zbc = np.zeros_like(rbc), np.zeros_like(zbs)
    if lasym:
        rbs[ntor, 2], zbc[ntor, 2] = .025, .035
    inp = vmex.VmecInput(
        mpol=4, ntor=ntor, nfp=2 if ntor else 1, ns_array=[9], ntheta=16,
        nzeta=8 if ntor else 0, niter_array=[3000], ftol_array=[1e-10],
        rbc=rbc, zbs=zbs, rbs=rbs, zbc=zbc, lasym=lasym,
        ai=[0.43], phiedge=6, delt=0.9,
    )
    eq = optimize.solve_equilibrium(inp)
    assert eq.result.converged
    return eq
