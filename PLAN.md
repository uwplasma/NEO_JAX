# NEO_JAX Port Plan (End-to-End JAX, JIT, Differentiable)

This plan is written as a step-by-step, always-on prompt. Follow it in order. Do not skip steps unless explicitly instructed. Keep each step small, testable, and reproducible.

## Current Status (Completed)
- Core NEO_JAX port complete with JAX scan backend, rational-surface correction, and parity on ORBITS_FAST, NCSX (fast + full gated), and LandremanPaul2021_QA_lowres.
- Streamed Fourier summation mode added for memory reduction; vectorized mode retained for speed.
- JAX scan body fused (neo_eval + RK4 + trapped update).
- User-friendly API added: `NeoConfig`, `run_neo`, `run_boozmn`, `run_boozer`, `run_booz_xform`, `NeoResults`, `build_surface_problem`.
- Surface selection supports normalized toroidal flux `s` (floats in [0, 1]) and index-based selections.
- Radial coordinates exposed (`s`, `sqrt_s`, `r_eff`) and plotting supports each as x-axis.
- Examples updated with full API usage, plotting, and SciPy least-squares autodiff demo over `Rmajor`.
- Documentation expanded: API guide, performance page, tutorials, validation.
- Parity fixtures tracked in `tests/fixtures/` with CI-ready fast cases.
- Added a JAX-native Boozer functional API in booz_xform_jax for end-to-end differentiation.
- Added in-memory vmec_jax → booz_xform_jax → neo_jax pipeline helper (`run_vmec_boozer_neo`)
  with a dedicated example and test (non-JIT end-to-end; uses WoutData).
- Packaging fixed for editable installs (explicit package list); docs build fixes applied (intersphinx, title underline).

## Immediate Next Steps (vmec_jax → booz_xform_jax → neo_jax)
1. **JAX-native Boozer adapter**:
   - Update `booz_xform_to_boozerdata` to preserve JAX arrays when inputs are JAX.
   - Eliminate `np.asarray` in the JAX path to keep JIT and gradients intact.
   - ✅ Added `vmec_jax.booz_xform_inputs_from_state` (JAX) to produce Boozer inputs
     without NumPy in the VMEC→Boozer path.
   - ✅ Added `booz_xform_to_boozerdata_jax` and surface metadata (`s_b`, `ns_b`, `jlist`) handling.
2. **JAX-native surface loop**:
   - Implement a `run_neo_jax` function that uses `jax.vmap` or `jax.lax.scan`
     over surfaces (no Python loop).
   - Return batched `NeoResults` with JAX arrays.
   - ✅ Added `run_neo_from_boozer_jax` (JAX surface scan) + `jax_surface_scan` flag.
   - 🔜 Add a conversion helper from JAX outputs to `NeoResults` (optional).
3. **JAX-safe surface initialization**:
   - Replace NumPy tie-breaker in `init_surface` for JIT path, or
     accept Bmin/Bmax/angles directly from `booz_xform_jax` to avoid
     host-side selection.
4. **Pipeline module**:
   - ✅ Added `neo_jax.pipeline.run_vmec_boozer_neo(vmec_source, neo_config)`:
     - accepts vmec_jax `FixedBoundaryRun`, `WoutData`, or input path.
     - runs vmec_jax → booz_xform_jax → neo_jax without file I/O.
   - 🔜 Upgrade to a **fully JIT** end-to-end path:
     - accept a JAX-native `vmec_state` and avoid NumPy in the VMEC→Boozer adapter.
     - return `NeoResults` with `jax.grad` support through the entire pipeline.
   - ✅ Added `build_vmec_boozer_neo_jax` to precompute Boozer constants and reuse a JAX callable.
5. **Gradient validation**:
   - Add a small gradient test: `grad(epsilon_effective)` w.r.t. a VMEC boundary
     coefficient at low resolution.
   - ✅ Added a forward-mode JVP gradient test (reverse-mode blocked by dynamic loops).
6. **Optimization demo**:
   - Provide an end-to-end optimization example that minimizes epsilon effective
     with respect to VMEC boundary coefficients.
   - ✅ Added QH warm-start optimization example (epsilon effective + aspect ratio).

## Required changes in booz_xform_jax
- Replace any NumPy usage with `jax.numpy` in core transforms.
- Remove any file I/O from the core; keep I/O in a thin wrapper only.
- Ensure all shapes are static and JIT-friendly (avoid dynamic list append).
- Replace Python loops with `lax.scan`/`vmap` where possible.
- Provide an explicit JAX API that returns a boozmn-like object with fields
  (`rmnc_b`, `zmns_b`, `pmns_b`, `bmnc_b`, `ixm_b`, `ixn_b`, `iota_b`,
  `buco_b`, `bvco_b`, `nfp_b`, `jlist`).

## Required changes in vmec_jax
- Provide a clean, pure JAX API that returns all fields needed by booz_xform_jax.
- Ensure no file I/O or side effects in the core VMEC solve.
- Expose boundary parameterization with JAX arrays for optimization.

## Deferred / Later
- GPU parity and overnight ORBITS full parity.
- Performance tuning on GPU after end-to-end JIT pipeline is ready.
- **Reverse-mode autodiff (long-term goal)**:
  - Refactor dynamic `lax.while_loop`/`fori_loop` paths in NEO to fixed-length
    `lax.scan` with static loop bounds (or introduce a `max_steps` scan with
    masking).
  - Isolate non-smooth trapped/passing event logic behind smoothing or
    differentiable surrogate logic where feasible.
  - Consider custom VJP for the field-line integration if scan-based reverse-mode
    remains too memory-intensive.
  - Add checkpointing and/or rematerialization to control reverse-mode memory.
  - Provide a dedicated reverse-mode validation test (compare to forward-mode).

## Goals
- Port the STELLOPT NEO Fortran code to a JAX implementation that reproduces `xneo` outputs and logs for the same inputs.
- Provide an end-to-end differentiable pipeline: `vmec_jax` -> `booz_xform_jax` -> `neo_jax`, fully JIT-friendly, CPU and GPU capable.
- Document everything in Read the Docs with equations, derivations, numerics, and source code references.
- Achieve tests with >90 percent coverage plus physics and numerical accuracy validation.

## Non-Goals
- Do not add new physics beyond NEO unless explicitly requested.
- Do not require file I/O for the core pipeline (file I/O only at the boundary for compatibility).

## Inputs and References
- STELLOPT NEO Fortran source: `/Users/rogerio/local/STELLOPT/NEO`
- VMEC and Boozer JAX sources: `vmec_jax` and `booz_xform_jax` from `github.com/uwplasma`
- Reference fixtures:
  - ORBITS outputs and debug arrays in `/Users/rogerio/local/tests/NEO_JAX/tests/fixtures/orbits`
  - NCSX example in `/Users/rogerio/local/tests/NEO_JAX/tests/fixtures/ncsx`

## Required Deliverables
- A `neo_jax` Python package.
- A CLI that matches `xneo` output and adds extra debug prints.
- A standalone Python driver script that runs the full pipeline without file I/O.
- Full docs (`docs/`, `conf.py`, `readthedocs.yml`) with equations and citations.
- Tests and benchmarks for CPU and GPU.

## Step 1: Map the Fortran Code (Authoritative Baseline)
1. Read and map the main flow in `/Users/rogerio/local/STELLOPT/NEO/Sources`:
   - `neo.f90` entry and control flow.
   - `neo_input`, `neo_init`, `neo_init_s`, `neo_fourier`, `neo_eval`.
   - Integration path: `flint_bo`, `rk4d_bo1`, `rhs_bo1`.
   - Root finding: `neo_zeros2d` for B min and max.
   - Splines: `spl2d`, `eva2d`, `eva2d_fd`, `eva2d_sd`, `poi2d`.
2. Write a precise mapping table: Fortran subroutine -> JAX function -> inputs -> outputs -> units.
3. Record all array shapes, index conventions, and normalization factors.
4. Record all diagnostics currently dumped in the Fortran build and how they are computed.

## Step 2: Define Data Models and Interfaces
1. Define immutable JAX-friendly dataclasses for:
   - `VmecData` (from vmec_jax).
   - `BoozerData` (from booz_xform_jax).
   - `NeoInputs` (control parameters and grids).
   - `NeoOutputs` (eps_eff, epspar, epstot, ctr, diagnostics).
2. All core functions must be pure and side-effect free:
   - No file I/O.
   - No global state.
   - Deterministic outputs for given inputs.
3. Distinguish static vs dynamic parameters for JIT:
   - Surface counts, M, N, grid sizes should be static.
   - Physics scalars can be dynamic.

## Step 3: vmec_jax Integration
1. Create a clean interface `vmec_to_boozer_inputs(vmec_state)` that extracts the arrays booz_xform_jax needs.
2. Ensure vmec_jax calls are pure and return JAX arrays only.
3. Provide a driver that supports:
   - Pure JAX execution (JIT).
   - Optional fallback to file I/O only in CLI compatibility mode.

## Step 4: booz_xform_jax JIT and Differentiability Plan
1. Make booz_xform_jax fully JIT compatible:
   - Replace NumPy with `jax.numpy` everywhere in core.
   - Remove file I/O and filesystem access from core routines.
   - Replace Python loops with `jax.lax.scan`, `jax.vmap`, or `jax.lax.fori_loop`.
   - Avoid dynamic shape creation at runtime. Shapes should be determined from static inputs.
   - Use `jax.lax.cond` for branching to keep JIT stable.
2. Provide a pure function:
   - `booz_xform_from_vmec(vmec_data, params) -> BoozerData`.
3. Provide optional precomputations:
   - Precompute trigonometric matrices using `vmap` so that they are re-used in JIT.
4. Ensure differentiation:
   - Test `jax.grad` on a small parameter (e.g., boundary coefficient) through Boozer output.
   - If numerical instabilities appear, add smoothing or solve with a differentiable root finder.
5. If needed, isolate non-differentiable operations (e.g., iterative solvers) behind `custom_jvp` or `custom_vjp`.

## Step 5: Implement NEO Core in JAX
1. Geometry evaluation:
   - Port Boozer metric computations from `neo_fourier`.
   - Confirm mapping to `b_tb`, `b_pb`, `gtbtb`, `gtbpb`, `gpbpb`, and related arrays.
2. Splines:
   - Implement 1D and 2D cubic splines in JAX.
   - Match the Fortran spline coefficients and boundary handling.
3. Root finding for B min and max:
   - Port `neo_zeros2d` to JAX.
   - Ensure stable convergence and differentiability.
4. Field line integration:
   - Implement `rhs_bo1` and `rk4d_bo1` in JAX.
   - Use `lax.scan` for stepping.
   - Preserve Fortran step sizing and rational surface handling.
5. Aggregate diagnostics:
   - Compute eps_eff, epspar, epstot, ctrone, ctrtot, and all existing diagnostics.
   - Produce optional debug arrays matching the Fortran dumps.

## Step 6: End-to-End Driver (Single Workflow)
1. Implement `neo_driver_jax(vmec_params, neo_params)`:
   - Run vmec_jax to get VMEC solution.
   - Convert to Boozer coordinates with booz_xform_jax.
   - Run neo_jax to compute eps_eff and diagnostics.
2. Provide:
   - `neo_jax.cli` for compatibility with `xneo`.
   - `neo_jax.run` for a pure Python driver (no files).
3. Use `jax.debug.print` with verbosity flags for extra prints in JIT.

## Step 7: Testing and Validation
1. Unit tests:
   - Splines vs Fortran coefficients.
   - Root finder vs Fortran `neo_zeros2d`.
   - RK4 integrator vs Fortran step output.
2. Golden tests:
   - Compare against ORBITS fixture arrays and `neo_out.ORBITS`.
   - Compare against NCSX example outputs.
3. Physics sanity tests:
   - Basic invariants from Boozer geometry.
   - Continuity of eps_eff vs surface index.
4. Performance tests:
   - CPU vs GPU timing for representative runs.
5. Coverage target:
   - Use `pytest-cov` and enforce >90 percent.

## Step 8: Documentation Plan
1. Docs structure:
   - `Overview`
   - `Installation`
   - `Quickstart`
   - `CLI Reference`
   - `Python API`
   - `Theory: NEO Equations and Derivations`
   - `Numerics: Splines, Root Finding, Integration`
   - `Boozer and VMEC Integration`
   - `Differentiability and JIT`
   - `Validation and Benchmarks`
   - `Tutorials: ORBITS, NCSX`
2. Include:
   - All content from the original NEO docs and tutorial.
   - Additional derivations and algorithm explanations.
   - Citations to relevant literature.
   - Source references back to Fortran modules and JAX code.
3. Provide equation-rich pages with explicit symbols, units, and normalization.

## Step 9: Acceptance Criteria
- End-to-end JAX pipeline produces eps_eff and diagnostics consistent with `xneo` for the ORBITS and NCSX fixtures.
- CLI matches `xneo` output plus optional verbose logs.
- JIT works for CPU and GPU, and `jax.grad` computes derivatives through the full pipeline.
- Documentation builds successfully on Read the Docs.
- Tests pass and coverage is above 90 percent.

## Step 10: Work Sequencing (Do This Order)
1. Repository scaffolding and docs skeleton.
2. Data model definitions.
3. Port splines and root finding.
4. Port geometry and integration.
5. Integrate booz_xform_jax and vmec_jax with JIT/diff.
6. Add end-to-end driver.
7. Add CLI and output matching.
8. Add tests and benchmarks.
9. Fill docs with equations, derivations, and tutorials.
