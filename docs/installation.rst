Installation
============

NEO_JAX is distributed as a standard Python package.

Base install
------------

Install from PyPI with:

.. code-block:: bash

   pip install neo-jax

For development from a clone, a standard editable install is:

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

VMEX→Boozer→NEO requires Python 3.11+, VMEX 0.11.6+ and booz_xform_jax 0.4.2+.
After the Boozer release, install:

.. code-block:: bash

   pip install "neo-jax[pipeline]"

Python 3.10 supports NEO_JAX without the optional VMEX adapter.

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
