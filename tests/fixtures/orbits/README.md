ORBITS reference fixture

Source:
- Generated locally from STELLOPT's NEO (`xneo`) and BOOZ_XFORM (`xbooz_xform`) using the BEAMS3D_TEST VMEC output.

Notes:
- `boozmn_ORBITS.nc` generated from `wout_ORBITS.nc` using `xbooz_xform`.
- `neo_in.ORBITS` is the control file used for `xneo ORBITS`.
- `neo_in.ORBITS_FAST` / `neo_out.ORBITS_FAST` are a reduced-resolution, reduced-surface fixture for fast parity checks.
- `boozmn_ORBITS_FAST.nc` is a copy of `boozmn_ORBITS.nc` to match the `ORBITS_FAST` extension expected by `xneo`.
- The `*_arr.dat` files are diagnostic arrays written by NEO when `write_output_files=1`; they correspond to the most recently processed surface (files are overwritten per surface).
- `conver.dat`, `diagnostic*.dat`, and `neolog.ORBITS` capture integration progress and diagnostics.
