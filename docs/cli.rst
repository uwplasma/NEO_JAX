CLI
===

NEO_JAX provides a legacy-compatible command-line interface for STELLOPT NEO.
The intent is that an existing ``xneo`` workflow can be pointed at the JAX
implementation and keep the same control-file and output-file conventions for
both the effective-ripple and parallel-current solves.

Installed entrypoints
---------------------

After ``pip install -e .``, the following commands are available:

.. code-block:: bash

   neo-jax
   xneo
   xneo_jax
   python -m neo_jax

The legacy form is:

.. code-block:: bash

   xneo [extension]

Examples:

.. code-block:: bash

   xneo ORBITS
   xneo_jax LandremanPaul2021_QA_lowres
   python -m neo_jax ORBITS_FAST

Control-file resolution
-----------------------

When an extension is supplied, NEO_JAX follows the same search order as the
STELLOPT executable:

1. ``neo_param.<extension>``
2. ``neo_param.in``
3. ``neo_in.<extension>``

If no extension is supplied, the CLI looks for ``neo.in``.

This behavior is implemented in ``neo_jax.io.resolve_control_path`` and is used
by the CLI by default. You can still override it explicitly with ``--control``.

Boozer input resolution
-----------------------

The input Boozer file depends on ``INP_SWI``:

- ``inp_swi = 0``:
  NEO_JAX follows the legacy ``boozmn_<extension>.nc`` convention used by
  ``xneo`` and STELLOPT's ``read_booz_in`` path.
- ``inp_swi != 0``:
  NEO_JAX resolves the path from ``IN_FILE`` in the control file.

You can override the automatic resolution with ``--boozmn``.

Legacy output files
-------------------

When the corresponding control switches are enabled, the CLI writes the same
legacy files that STELLOPT writes:

- main output: ``neo_out.*``
- parallel-current summary: ``neo_cur.*`` when ``calc_cur = 1``
- log: ``neolog.*``
- diagnostics:
  ``diagnostic.dat``, ``diagnostic_add.dat``, ``diagnostic_bigint.dat``
- convergence history: ``conver.dat``
- current-history dump: ``current.dat`` when ``write_cur_inte = 1``
- geometry / Fourier dumps:
  ``dimension.dat``, ``theta_arr.dat``, ``phi_arr.dat``, ``mn_arr.dat``,
  ``rmnc_arr.dat``, ``zmns_arr.dat``, ``lmns_arr.dat``, ``bmnc_arr.dat``,
  ``b_s_arr.dat``, ``r_s_arr.dat``, ``z_s_arr.dat``, ``l_s_arr.dat``,
  ``isqrg_arr.dat``, ``sqrg11_arr.dat``, ``kg_arr.dat``, ``pard_arr.dat``,
  ``r_tb_arr.dat``, ``z_tb_arr.dat``, ``p_tb_arr.dat``, ``b_tb_arr.dat``,
  ``r_pb_arr.dat``, ``z_pb_arr.dat``, ``p_pb_arr.dat``, ``b_pb_arr.dat``,
  ``gtbtb_arr.dat``, ``gpbpb_arr.dat``, ``gtbpb_arr.dat``

The Fortran-style formatting used in the text files is implemented in
``neo_jax.legacy`` so that ``neo_out.*``, ``neolog.*``, ``diagnostic*.dat``,
``neo_cur.*``, and ``conver.dat`` match the STELLOPT text output exactly on the
parity cases. ``current.dat`` follows the same token layout and special-value
formatting (including gfortran's omitted exponent letter for some 3-digit
exponents) and is validated numerically token-by-token against the reference
binary.

Compatibility scope
-------------------

Supported legacy scope:

- ``calc_cur = 0`` effective-ripple runs
- ``calc_cur = 1`` parallel-current runs
- control-file precedence:
  ``neo_param.<extension>`` -> ``neo_param.in`` -> ``neo_in.<extension>``
- legacy Boozer input naming via ``boozmn_<extension>.nc`` for ``inp_swi = 0``

Examples
--------

Using the shipped ORBITS fixture:

.. code-block:: bash

   cd tests/fixtures/orbits
   python -m neo_jax ORBITS_FAST

Using the dense Landreman/Paul QA fixture:

.. code-block:: bash

   cd tests/fixtures/landreman_qa_lowres
   xneo LandremanPaul2021_QA_lowres

Using a current-enabled ORBITS case:

.. code-block:: bash

   cd /path/to/case
   xneo ORBITS_CURINT
   xneo_jax ORBITS_CURINT

Both commands will read the same control file and write the same
``neo_out.*`` / ``neo_cur.*`` outputs.

Optional arguments
------------------

NEO_JAX keeps the legacy positional interface but also exposes a few explicit
overrides:

- ``--control``: use a specific control file path
- ``--boozmn``: use a specific ``boozmn`` path
- ``--output``: override the main output file name
- ``--jax``: prefer the JAX backend when compatible
- ``--no-jax``: force the Python backend
- ``--verbose``: print extra progress information

Note that control-file ``WRITE_PROGRESS`` is honored by default, so legacy runs
still print progress when the control file requests it.

Testing
-------

The CLI parity tests live in ``tests/regression/test_cli_legacy.py``. They run
the real STELLOPT executable from ``~/bin/xneo`` (or ``NEO_REFERENCE_BIN``)
against the JAX CLI on multiple geometries and control-file layouts:

- ``LandremanPaul2021_QA_lowres`` dense legacy fixture
- synthetic ORBITS single-surface case with diagnostics and array dumps
- synthetic ORBITS ``calc_cur = 1`` case with ``current.dat``
- NCSX mini case
- control-file precedence checks for ``neo_param.*`` vs ``neo_in.*``

Additional full-fixture ORBITS and NCSX CLI parity tests are available behind
``NEO_JAX_RUN_SLOW=1``.
