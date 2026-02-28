CLI
===

The CLI mirrors ``xneo`` conventions while providing additional verbosity.

Basic usage:

.. code-block:: bash

   neo-jax [extension]

The ``extension`` argument follows the ``xneo`` search order for control files:
``neo_param.<ext>``, then ``neo_param.in``, then ``neo_in.<ext>``. If no
extension is supplied, ``neo.in`` is used.

Useful options:

- ``--boozmn``: path to a ``boozmn`` file (netCDF).
- ``--output``: override the NEO output file name.
- ``--jax``: use the JAX backend (default).
- ``--no-jax``: disable the JAX backend.
- ``--verbose``: print progress and output lines to stdout.

Example using the ORBITS reference case:

.. code-block:: bash

   cd tests/fixtures/orbits
   neo-jax ORBITS --boozmn boozmn_ORBITS.nc --verbose

Output format is controlled by ``eout_swi`` in the control file, matching the
original NEO conventions.
