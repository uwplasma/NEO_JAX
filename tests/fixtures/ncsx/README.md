NCSX reference fixture

Source:
- NEO tutorial example (Boozer file and control-file example). See the NEO tutorial page for context and parameter meanings.

Notes:
- `boozmn_ncsx_c09r00_free.nc` is the Boozer file used by the tutorial example.
- `neo_in.ncsx_c09r00_free` mirrors the tutorial control-file layout (with the first four lines ignored by NEO when `inp_swi=0`).
- `neo_in.ncsx_c09r00_free_fast` is a reduced test (4 surfaces, 64x64 grid) for CI parity.
- The control files use the extension `ncsx_c09r00_free` so NEO resolves `boozmn_ncsx_c09r00_free.nc` when running `xneo ncsx_c09r00_free`.

Regeneration (fast reference):

```bash
# Temporarily use the fast control file as neo_param.in so xneo reads it
cp neo_in.ncsx_c09r00_free_fast neo_param.in
/Users/rogerio/local/STELLOPT/NEO/Release/xneo ncsx_c09r00_free
rm neo_param.in
```
