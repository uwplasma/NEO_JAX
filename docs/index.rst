NEO_JAX
=======

NEO_JAX is a JAX port of the STELLOPT NEO code for computing effective helical
ripple and related neoclassical transport diagnostics in Boozer coordinates.

Legacy CLI parity
-----------------

NEO_JAX also ships an ``xneo``-compatible CLI. The same ``neo_in.*`` /
``neo_param.*`` control files used by STELLOPT can be run with ``xneo_jax`` or
``python -m neo_jax`` while keeping the legacy file naming conventions
(``neo_out.*``, ``neo_cur.*``, ``neolog.*``, ``diagnostic*.dat``,
``current.dat``, and the auxiliary ``*_arr.dat`` dumps). See :doc:`cli` and
:doc:`validation` for the exact compatibility contract, GPU smoke coverage,
and parity limits on the dense legacy fixtures.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   overview
   installation
   quickstart
   user_api
   cli
   vmec_boozer
   theory
   numerics
   differentiability
   tutorials_orbits
   tutorials_ncsx
   validation
   performance
   fortran_map
   api
   references
