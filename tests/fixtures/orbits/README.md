ORBITS reference fixture

Source:
- Generated locally from STELLOPT's NEO (`xneo`) and BOOZ_XFORM (`xbooz_xform`) using the BEAMS3D_TEST VMEC output.

Notes:
- `boozmn_ORBITS.nc` generated from `wout_ORBITS.nc` using `xbooz_xform`.
- `neo_in.ORBITS` is the control file used for `xneo ORBITS`.
- The `*_arr.dat` files are diagnostic arrays written by NEO when `write_output_files=1`; they correspond to the most recently processed surface (files are overwritten per surface).
- `conver.dat`, `diagnostic*.dat`, and `neolog.ORBITS` capture integration progress and diagnostics.
