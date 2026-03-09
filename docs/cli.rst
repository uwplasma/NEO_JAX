CLI
===

NEO_JAX provides a legacy-compatible command-line interface for STELLOPT NEO.
The intent is that an existing ``xneo`` workflow can be pointed at the JAX
implementation and keep the same control-file and output-file conventions for
the effective-ripple solve.

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
- log: ``neolog.*``
- diagnostics:
  ``diagnostic.dat``, ``diagnostic_add.dat``, ``diagnostic_bigint.dat``
- convergence history: ``conver.dat``
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
and ``conver.dat`` match the STELLOPT text output exactly on the parity cases.

Compatibility scope
-------------------

The legacy CLI currently targets the effective-ripple solve:

- supported: ``calc_cur = 0``
- not yet supported: ``calc_cur = 1`` (parallel-current output path)

If ``calc_cur`` is enabled, the CLI raises ``NotImplementedError`` rather than
quietly producing incomplete output.

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
