#!/usr/bin/env python3
"""Optimize epsilon effective + aspect ratio starting from a QH warm start.

This example mirrors the spirit of simsopt's 2_Intermediate QH optimization:
we start from the QH warm-start VMEC input and vary a small set of boundary
Fourier coefficients. The objective combines epsilon effective from NEO_JAX
with a penalty on the aspect ratio, computed directly from the boundary shape.

Note: NEO_JAX's JAX surface scan currently supports forward-mode autodiff.
We therefore use ``jax.jacfwd`` for gradients in this example.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _parse_surfaces(text: str) -> list[float | int]:
    items: list[float | int] = []
    for raw in text.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if any(ch in raw for ch in (".", "e", "E")):
            items.append(float(raw))
        else:
            items.append(int(raw))
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt-steps", type=int, default=2, help="Outer optimization steps.")
    parser.add_argument("--opt-lr", type=float, default=5e-3, help="Gradient descent step size.")
    parser.add_argument("--max-iter", type=int, default=2, help="VMEC iterations per objective eval.")
    parser.add_argument("--step-size", type=float, default=5e-3, help="VMEC update step size.")
    parser.add_argument(
        "--surfaces",
        type=str,
        default="0.3,0.6,0.8",
        help="Comma-separated s values (0-1) or 1-based indices.",
    )
    parser.add_argument("--theta-n", type=int, default=16, help="NEO theta grid size.")
    parser.add_argument("--phi-n", type=int, default=16, help="NEO phi grid size.")
    parser.add_argument("--mboz", type=int, default=8, help="Boozer m resolution.")
    parser.add_argument("--nboz", type=int, default=8, help="Boozer n resolution.")
    parser.add_argument("--aspect-weight", type=float, default=1.0, help="Aspect ratio penalty weight.")
    parser.add_argument(
        "--aspect-target",
        type=float,
        default=None,
        help="Target aspect ratio (default: baseline from warm start).",
    )
    parser.add_argument("--jit-booz", action="store_true", help="JIT the Boozer transform.")
    args = parser.parse_args()

    import vmec_jax as vj
    from vmec_jax._compat import enable_x64, has_jax, jax, jnp
    from vmec_jax.booz_input import booz_xform_inputs_from_state
    from vmec_jax.boundary import boundary_from_indata
    from vmec_jax.driver import example_paths
    from vmec_jax.fourier import build_helical_basis, eval_fourier
    from vmec_jax.init_guess import initial_guess_from_boundary
    from vmec_jax.static import build_static

    from booz_xform_jax.jax_api import prepare_booz_xform_constants, booz_xform_jax_impl
    from neo_jax import NeoConfig
    from neo_jax.driver import run_neo_from_boozer_jax
    from neo_jax.io import booz_xform_to_boozerdata

    if not has_jax():
        raise SystemExit("This example requires JAX (pip install -e '.[jax]').")
    enable_x64(True)

    input_path, _ = example_paths("nfp4_QH_warm_start")
    cfg, indata = vj.load_config(input_path)
    static = build_static(cfg)
    boundary0 = boundary_from_indata(indata, static.modes)

    modes = static.modes

    def _mode_index(m: int, n: int) -> int:
        mask = (np.asarray(modes.m) == m) & (np.asarray(modes.n) == n)
        idx = np.where(mask)[0]
        if idx.size == 0:
            raise ValueError(f"Mode (m={m}, n={n}) not found in VMEC mode table")
        return int(idx[0])

    # Small set of boundary coefficients to vary.
    k10 = _mode_index(1, 0)
    k11 = _mode_index(1, 1)
    param_names = ["dRcos(1,0)", "dZsin(1,0)", "dRcos(1,1)", "dZsin(1,1)"]

    st0 = initial_guess_from_boundary(static, boundary0, indata, vmec_project=False)
    g0 = vj.eval_geom(st0, static)
    signgs0 = vj.signgs_from_sqrtg(np.asarray(g0.sqrtg), axis_index=1)
    flux = vj.flux_profiles_from_indata(indata, static.s, signgs=signgs0)
    pressure = jnp.zeros_like(jnp.asarray(static.s))

    base_inputs = booz_xform_inputs_from_state(
        state=st0,
        static=static,
        indata=indata,
        signgs=signgs0,
    )

    constants, grids = prepare_booz_xform_constants(
        nfp=int(base_inputs.nfp),
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        asym=bool(cfg.lasym),
        xm=np.asarray(base_inputs.xm),
        xn=np.asarray(base_inputs.xn),
        xm_nyq=np.asarray(base_inputs.xm_nyq),
        xn_nyq=np.asarray(base_inputs.xn_nyq),
    )

    s_half = 0.5 * (np.asarray(static.s[:-1]) + np.asarray(static.s[1:]))
    surface_requests = _parse_surfaces(args.surfaces)
    surface_indices: list[int] = []
    for val in surface_requests:
        if isinstance(val, float) and 0.0 <= val <= 1.0:
            surface_indices.append(int(np.argmin(np.abs(s_half - val))))
        else:
            surface_indices.append(int(val) - 1)
    surface_indices_j = jnp.asarray(surface_indices, dtype=jnp.int32)
    selected_s = s_half[np.asarray(surface_indices)]

    neo_config = NeoConfig(
        surfaces=None,
        theta_n=int(args.theta_n),
        phi_n=int(args.phi_n),
        npart=12,
        multra=1,
        nstep_per=6,
        nstep_min=30,
        nstep_max=60,
        no_bins=20,
        acc_req=0.1,
    )
    control = neo_config.to_control()

    basis = build_helical_basis(static.modes, static.grid)

    def boundary_aspect_ratio(boundary) -> jnp.ndarray:
        Rb = eval_fourier(jnp.asarray(boundary.R_cos), jnp.asarray(boundary.R_sin), basis)
        Zb = eval_fourier(jnp.asarray(boundary.Z_cos), jnp.asarray(boundary.Z_sin), basis)
        dA = Rb * jnp.roll(Zb, -1, axis=0) - jnp.roll(Rb, -1, axis=0) * Zb
        area = 0.5 * jnp.sum(dA, axis=0)
        minor = jnp.sqrt(jnp.abs(area) / jnp.pi)
        Rmax = jnp.max(Rb, axis=0)
        Rmin = jnp.min(Rb, axis=0)
        Rmajor = jnp.mean(0.5 * (Rmax + Rmin))
        Aminor = jnp.mean(minor)
        return Rmajor / Aminor

    aspect0 = float(boundary_aspect_ratio(boundary0))
    aspect_target = float(args.aspect_target) if args.aspect_target is not None else aspect0

    booz_fn = booz_xform_jax_impl
    if args.jit_booz:
        booz_fn = jax.jit(booz_xform_jax_impl, static_argnames=("constants",))

    def _build_boundary(params):
        dR10, dZ10, dR11, dZ11 = params
        Rcos = jnp.asarray(boundary0.R_cos).at[k10].add(dR10).at[k11].add(dR11)
        Zsin = jnp.asarray(boundary0.Z_sin).at[k10].add(dZ10).at[k11].add(dZ11)
        return vj.BoundaryCoeffs(
            R_cos=Rcos,
            R_sin=jnp.asarray(boundary0.R_sin),
            Z_cos=jnp.asarray(boundary0.Z_cos),
            Z_sin=Zsin,
        )

    def _solve_vmec(boundary):
        st_guess = initial_guess_from_boundary(static, boundary, indata, vmec_project=False)
        res = vj.solve_fixed_boundary_gd(
            st_guess,
            static,
            phipf=flux.phipf,
            chipf=flux.chipf,
            signgs=signgs0,
            lamscale=flux.lamscale,
            pressure=pressure,
            gamma=float(indata.get_float("GAMMA", 0.0)),
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            jacobian_penalty=1e3,
            jit_grad=False,
            differentiable=True,
            stop_grad_in_update=True,
            verbose=False,
        )
        return res.state

    def epsilon_effective_from_state(state):
        inputs = booz_xform_inputs_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=signgs0,
        )
        booz_out = booz_fn(
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
            surface_indices=surface_indices_j,
        )
        booz = booz_xform_to_boozerdata(booz_out, use_jax=True)
        outputs = run_neo_from_boozer_jax(booz, control)
        return jnp.mean(outputs.eps_eff)

    def objective(params):
        boundary = _build_boundary(params)
        state = _solve_vmec(boundary)
        eps_eff = epsilon_effective_from_state(state)
        aspect = boundary_aspect_ratio(boundary)
        return eps_eff + float(args.aspect_weight) * (aspect - aspect_target) ** 2

    def metrics(params):
        boundary = _build_boundary(params)
        state = _solve_vmec(boundary)
        eps_eff = epsilon_effective_from_state(state)
        aspect = boundary_aspect_ratio(boundary)
        obj = eps_eff + float(args.aspect_weight) * (aspect - aspect_target) ** 2
        return obj, eps_eff, aspect

    params = jnp.zeros((4,), dtype=jnp.float64)
    obj0, eps0, aspect_init = metrics(params)
    print("NEO_JAX QH optimization (epsilon effective + aspect ratio)")
    print("Surfaces (s):", ", ".join(f"{s:.3f}" for s in selected_s))
    print("Params:", ", ".join(param_names))
    print(f"Baseline eps_eff={float(eps0):.6e}, aspect={float(aspect_init):.4f}")

    grad_fn = jax.jacfwd(objective)

    for step in range(int(args.opt_steps)):
        obj, eps_eff, aspect = metrics(params)
        grad = grad_fn(params)
        params = params - float(args.opt_lr) * grad
        print(
            f"step {step:02d}: obj={float(obj):.6e} "
            f"eps_eff={float(eps_eff):.6e} aspect={float(aspect):.4f}"
        )

    print("final params:", np.asarray(params))
    print("target aspect:", aspect_target)


if __name__ == "__main__":
    main()
