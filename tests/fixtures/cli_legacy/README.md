CLI legacy reference fixtures
=============================

These files are frozen reference outputs generated with the local STELLOPT
``xneo`` executable and committed so the default NEO_JAX test suite can verify
CLI parity without needing that external binary at test time.

Cases included here:

- ``orbits_mini``: one-surface ORBITS case covering ``neo_out.*``,
  ``neolog.*``, ``diagnostic*.dat``, ``conver.dat``, and the legacy geometry
  dumps.
- ``orbits_curint``: one-surface ORBITS ``calc_cur = 1`` case covering
  ``neo_cur.*`` and ``current.dat``.
- ``ncsx_mini``: one-surface NCSX case covering ``neo_out.*`` and
  ``neolog.*``.

The corresponding Boozer inputs are reused from the main fixture directories:

- ``tests/fixtures/orbits/boozmn_ORBITS_FAST.nc``
- ``tests/fixtures/ncsx/boozmn_ncsx_c09r00_free.nc`` (resolved from the
  external fixture cache in slim checkouts)
