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
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

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
    parse_surface_list,
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


def main() -> None:
    parser = argparse.ArgumentParser()
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
        "--param-scale-min",
        type=float,
        default=1.0e-3,
        help="Minimum scale (meters) applied to boundary parameter updates.",
    )
    parser.add_argument(
        "--aspect-target",
        type=float,
        default=None,
        help="Target aspect ratio (default: baseline from warm start).",
    )
    parser.add_argument("--jit-booz", action="store_true", help="JIT the Boozer transform.")
    parser.add_argument("--jit-residual", action="store_true", help="JIT residual and Jacobian.")
    parser.add_argument(
        "--print-jacobian",
        action="store_true",
        help="Print Jacobian statistics before optimization.",
    )
    parser.add_argument(
        "--least-squares-max-nfev",
        type=int,
        default=10,
        help="Max function evaluations for SciPy least squares.",
    )
    args = parser.parse_args()

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
    param_scales = []
    for spec in param_specs:
        if spec.kind == "rc":
            base = float(np.asarray(boundary0.R_cos)[spec.index])
        elif spec.kind == "rs":
            base = float(np.asarray(boundary0.R_sin)[spec.index])
        elif spec.kind == "zc":
            base = float(np.asarray(boundary0.Z_cos)[spec.index])
        else:
            base = float(np.asarray(boundary0.Z_sin)[spec.index])
        scale = max(abs(base), float(args.param_scale_min))
        param_scales.append(scale)
    param_scales = jnp.asarray(param_scales, dtype=jnp.float64)

    ctx = prepare_fixed_boundary_context(static=static, indata=indata, boundary=boundary0)

    constants, grids = prepare_booz_xform_constants_from_inputs(
        inputs=ctx.booz_inputs,
        mboz=int(args.mboz),
        nboz=int(args.nboz),
        asym=bool(cfg.lasym),
    )

    surface_requests = parse_surface_list(args.surfaces)
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
        return apply_boundary_params(boundary0, param_specs, params * param_scales)

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
            stop_grad_in_update=False,
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
        return outputs.eps_eff

    def residual_vector(params):
        boundary = _build_boundary(params)
        state = _solve_vmec(boundary)
        eps_eff = epsilon_effective_from_state(state)
        aspect = boundary_aspect_ratio(boundary, basis)
        res = eps_eff
        if float(args.aspect_weight) > 0.0:
            aspect_res = jnp.sqrt(float(args.aspect_weight)) * (aspect - aspect_target)
            res = jnp.concatenate([eps_eff, jnp.atleast_1d(aspect_res)])
        return res

    params = jnp.zeros((len(param_specs),), dtype=jnp.float64)
    print("NEO_JAX QH optimization (epsilon effective + aspect ratio)")
    print("Surfaces (s):", ", ".join(f"{s:.3f}" for s in selected_s))
    print("Params:", ", ".join(param_names))
    print("Param scales:", ", ".join(f"{float(s):.2e}" for s in np.asarray(param_scales)))
    print("Fixed:", ", ".join(fixed) if fixed else "none")
    print(f"Max mode: {int(args.max_mode)}")
    print(f"Param threshold |rc/zs| > {float(args.min_coeff):.1e}")
    print("  [info] evaluating baseline (first call may JIT-compile) ...")
    t0 = time.perf_counter()
    baseline_state = _solve_vmec(boundary0)
    eps0 = np.asarray(epsilon_effective_from_state(baseline_state), dtype=float)
    aspect_init = float(boundary_aspect_ratio(boundary0, basis))
    t1 = time.perf_counter()
    print(f"  [info] baseline evaluation took {t1 - t0:.2f}s")
    print(f"Baseline eps_eff={float(np.mean(eps0)):.6e}, aspect={float(aspect_init):.4f}")
    print("  [info] Warm-start is often near-optimal; increase --param-scale-min or --max-mode for larger updates.")

    residual_fn = residual_vector
    jac_fn = jax.jacfwd(residual_vector)
    if bool(args.jit_residual):
        residual_fn = jax.jit(residual_vector)
        jac_fn = jax.jit(jac_fn)
        print("  [info] JIT-compiling residual and Jacobian ...")
        t0 = time.perf_counter()
        _ = jax.block_until_ready(residual_fn(params))
        _ = jax.block_until_ready(jac_fn(params))
        t1 = time.perf_counter()
        print(f"  [info] JIT compile done in {t1 - t0:.2f}s")

    def residual_np(p: np.ndarray) -> np.ndarray:
        res = residual_fn(jnp.asarray(p, dtype=jnp.float64))
        res = np.asarray(res, dtype=float)
        if not np.all(np.isfinite(res)):
            print("  [warn] non-finite residual; returning large penalty.")
            res = np.nan_to_num(res, nan=1e6, posinf=1e6, neginf=-1e6)
        return res

    def jac_np(p: np.ndarray) -> np.ndarray:
        jac = jac_fn(jnp.asarray(p, dtype=jnp.float64))
        jac = np.asarray(jac, dtype=float)
        if not np.all(np.isfinite(jac)):
            print("  [warn] non-finite Jacobian; returning zeros.")
            jac = np.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
        return jac

    if bool(args.print_jacobian):
        jac0 = jac_np(np.asarray(params, dtype=float))
        res0 = residual_np(np.asarray(params, dtype=float))
        print(
            "  [info] Jacobian stats:",
            f"norm={np.linalg.norm(jac0):.3e}",
            f"max={np.max(np.abs(jac0)):.3e}",
        )
        try:
            dx, *_ = np.linalg.lstsq(jac0, -res0, rcond=None)
            print(f"  [info] linearized step norm={np.linalg.norm(dx):.3e}")
        except Exception:
            print("  [warn] linearized step solve failed.")

    print("  [info] starting SciPy least_squares optimization ...")
    result = least_squares(
        residual_np,
        x0=np.asarray(params),
        jac=jac_np,
        verbose=2,
        max_nfev=int(args.least_squares_max_nfev),
        x_scale="jac",
    )

    print("optimizer status:", result.message)
    print(
        "final params:",
        np.array2string(np.asarray(result.x), precision=6, floatmode="maxprec_equal"),
    )
    print("target aspect:", aspect_target)
    final_res = residual_fn(jnp.asarray(result.x, dtype=jnp.float64))
    final_res = np.asarray(final_res, dtype=float)
    eps_final = final_res[: len(selected_s)]
    print(f"final eps_eff mean={float(np.mean(eps_final)):.6e}")
    if float(args.aspect_weight) > 0.0:
        aspect_final = aspect_target + final_res[-1] / np.sqrt(float(args.aspect_weight))
        print(f"final aspect={float(aspect_final):.4f}")


if __name__ == "__main__":
    main()
