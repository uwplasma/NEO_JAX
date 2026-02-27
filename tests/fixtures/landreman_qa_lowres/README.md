# LandremanPaul2021_QA_lowres fixture

This fixture was generated from the VMEC input file
`input.LandremanPaul2021_QA_lowres` (from `simsopt/tests/test_files`).
The data flow is:

1. VMEC (`xvmec2000`) to produce `wout_LandremanPaul2021_QA_lowres.nc`.
2. Boozer transform (`xbooz_xform`) to produce `boozmn_LandremanPaul2021_QA_lowres.nc`.
3. NEO (`xneo`) to produce `neo_out.LandremanPaul2021_QA_lowres`.

Regeneration commands (from this directory):

```bash
# VMEC
/Users/rogerio/local/STELLOPT/VMEC2000/Release/xvmec2000 input.LandremanPaul2021_QA_lowres noscreen

# Boozer transform (surfaces 38 and 50)
cat > input.boz_LandremanPaul2021_QA_lowres <<'BOZ'
32 32
LandremanPaul2021_QA_lowres
38 50
BOZ
/Users/rogerio/local/STELLOPT/BOOZ_XFORM/Release/xbooz_xform input.boz_LandremanPaul2021_QA_lowres F

# NEO reference output
/Users/rogerio/local/STELLOPT/NEO/Release/xneo LandremanPaul2021_QA_lowres
```

Notes:
- `neo_in.LandremanPaul2021_QA_lowres` uses two surfaces (38, 50) for a fast
  parity test.
- Intermediate VMEC outputs (wout, threed1, mercier, etc.) are not tracked.
