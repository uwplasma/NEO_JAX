#!/usr/bin/env python3
"""Optimize epsilon effective + aspect ratio starting from a QH warm start.

This example mirrors the spirit of simsopt's 2_Intermediate QH optimization:
we start from the QH warm-start VMEC input and vary a small set of boundary
Fourier coefficients. The objective combines epsilon effective from NEO_JAX
with a penalty on the aspect ratio, computed directly from the boundary shape.

Note: NEO_JAX's JAX surface scan currently supports forward-mode autodiff.
We therefore use ``jax.jacfwd`` for gradients in this example.
We fix rc00 (R_cos at m=0,n=0) so the major radius stays constant.
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
        "--max-mode",
        type=int,
        default=1,
        help="Optimize all boundary modes with |m|,|n| <= max-mode.",
    )
    parser.add_argument(
        "--fix",
        type=str,
        default="rc00",
        help="Comma-separated list of fixed parameters (e.g. rc00, zs10).",
    )
    parser.add_argument(
        "--min-coeff",
        type=float,
        default=0.0,
        help="Minimum |rc/zs| to include as an optimization parameter (0 includes all).",
    )
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
    from vmec_jax.boundary import boundary_from_indata, boundary_aspect_ratio
    from vmec_jax.driver import example_paths
    from vmec_jax.driver import solve_fixed_boundary_from_boundary
    from vmec_jax.fourier import build_helical_basis
    from vmec_jax.optimization import (
        apply_boundary_params,
        boundary_param_names,
        boundary_param_specs,
        prepare_fixed_boundary_context,
        surface_indices_from_static,
    )
    from vmec_jax.static import build_static

    from booz_xform_jax.jax_api import prepare_booz_xform_constants_from_inputs, booz_xform_from_inputs
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
    basis = build_helical_basis(static.modes, static.grid)

    fixed = [item.strip() for item in args.fix.split(",") if item.strip()]

    param_specs = boundary_param_specs(
        boundary0,
        static.modes,
        max_mode=int(args.max_mode),
        min_coeff=float(args.min_coeff),
        include=("rc", "zs"),
        fix=fixed,
    )
    if not param_specs:
        raise RuntimeError("No nonzero rc/zs modes found to optimize.")

    param_names = boundary_param_names(param_specs)

    ctx = prepare_fixed_boundary_context(static=static, indata=indata, boundary=boundary0)

    constants, grids = prepare_booz_xform_constants_from_inputs(
        inputs=ctx.booz_inputs,
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        asym=bool(cfg.lasym),
    )

    surface_requests = _parse_surfaces(args.surfaces)
    surface_indices, selected_s = surface_indices_from_static(static, surface_requests)
    surface_indices_j = jnp.asarray(surface_indices, dtype=jnp.int32)

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

    aspect0 = float(boundary_aspect_ratio(boundary0, basis))
    aspect_target = float(args.aspect_target) if args.aspect_target is not None else aspect0

    def _build_boundary(params):
        return apply_boundary_params(boundary0, param_specs, params)

    def _solve_vmec(boundary):
        return solve_fixed_boundary_from_boundary(
            boundary=boundary,
            static=static,
            indata=indata,
            flux=ctx.flux,
            pressure=ctx.pressure,
            signgs=ctx.signgs,
            max_iter=int(args.max_iter),
            step_size=float(args.step_size),
            jacobian_penalty=1e3,
            jit_grad=False,
            differentiable=True,
            stop_grad_in_update=True,
            verbose=False,
        )

    def epsilon_effective_from_state(state):
        inputs = booz_xform_inputs_from_state(
            state=state,
            static=static,
            indata=indata,
            signgs=ctx.signgs,
        )
        booz_out = booz_xform_from_inputs(
            inputs=inputs,
            constants=constants,
            grids=grids,
            surface_indices=surface_indices_j,
            jit=bool(args.jit_booz),
        )
        booz = booz_xform_to_boozerdata(booz_out, use_jax=True)
        outputs = run_neo_from_boozer_jax(booz, control)
        return jnp.mean(outputs.eps_eff)

    def objective(params):
        boundary = _build_boundary(params)
        state = _solve_vmec(boundary)
        eps_eff = epsilon_effective_from_state(state)
        aspect = boundary_aspect_ratio(boundary, basis)
        return eps_eff + float(args.aspect_weight) * (aspect - aspect_target) ** 2

    def metrics(params):
        boundary = _build_boundary(params)
        state = _solve_vmec(boundary)
        eps_eff = epsilon_effective_from_state(state)
        aspect = boundary_aspect_ratio(boundary, basis)
        obj = eps_eff + float(args.aspect_weight) * (aspect - aspect_target) ** 2
        return obj, eps_eff, aspect

    params = jnp.zeros((len(param_specs),), dtype=jnp.float64)
    obj0, eps0, aspect_init = metrics(params)
    print("NEO_JAX QH optimization (epsilon effective + aspect ratio)")
    print("Surfaces (s):", ", ".join(f"{s:.3f}" for s in selected_s))
    print("Params:", ", ".join(param_names))
    print("Fixed:", ", ".join(fixed) if fixed else "none")
    print(f"Max mode: {int(args.max_mode)}")
    print(f"Param threshold |rc/zs| > {float(args.min_coeff):.1e}")
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
