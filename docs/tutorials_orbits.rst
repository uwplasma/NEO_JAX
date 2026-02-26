Tutorial: ORBITS Reference
==========================

The ORBITS reference case is included in ``tests/fixtures/orbits`` and provides
``boozmn_ORBITS.nc`` plus the reference ``neo_out.ORBITS`` output.

Run via CLI
-----------

.. code-block:: bash

   cd tests/fixtures/orbits
   neo-jax ORBITS --boozmn boozmn_ORBITS.nc --verbose

Expected output lines should match ``neo_out.ORBITS`` (within numerical
precision). For example, the first line of the reference output is:

.. code-block:: text

         2  0.1345303742E-07  0.1256737995E-01  0.9990079980E+00  0.5019329311E+01  0.1002132101E+02

Run via Python
--------------

.. code-block:: python

   from neo_jax.control import read_control
   from neo_jax.driver import run_neo_from_boozmn

   control = read_control("tests/fixtures/orbits/neo_in.ORBITS")
   results = run_neo_from_boozmn("tests/fixtures/orbits/boozmn_ORBITS.nc", control)
   print(results[0])

Debug outputs
-------------

When ``write_output_files`` is enabled in the control file, NEO_JAX can write
intermediate arrays (``b_s_arr.dat``, ``r_s_arr.dat``, etc.) that are included
in the fixture for regression testing.
