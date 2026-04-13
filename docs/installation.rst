Installation
============

NEO_JAX is distributed as a standard Python package.

Base install
------------

A standard editable install is:

.. code-block:: bash

   cd NEO_JAX
   pip install -e .

Development and documentation extras
------------------------------------

Optional development and documentation dependencies are installed with:

.. code-block:: bash

   pip install -e ".[dev,docs]"

JAX should be installed with the correct accelerator support for your system
(CPU, CUDA, or ROCm). Consult the `JAX installation guide <https://docs.jax.dev/en/latest/installation.html>`_
for platform-specific instructions.

The ``boozmn`` reader relies on the ``netCDF4`` Python package, which is listed
as a core dependency.

Optional pipeline dependencies
------------------------------

For end-to-end VMEC→Boozer→NEO workflows, install:

- `vmec_jax <https://github.com/uwplasma/vmec_jax>`_
- `booz_xform_jax <https://github.com/uwplasma/booz_xform_jax>`_

The CI workflow installs both packages for the pipeline tests.

Building the documentation
--------------------------

To build the documentation locally:

.. code-block:: bash

   python -m sphinx -b html docs docs/_build/html

For a fast structural check without generating the full HTML tree:

.. code-block:: bash

   python -m sphinx -b dummy docs docs/_build/dummy

Continuous integration
----------------------

The GitHub Actions workflow installs the package, runs the test suite on CPU,
and executes a small performance regression check. See :doc:`testing` for the
full workflow.
