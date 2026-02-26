Installation
============

NEO_JAX is a Python package. A standard editable install is:

.. code-block:: bash

   cd NEO_JAX
   pip install -e .

Optional development and documentation dependencies:

.. code-block:: bash

   pip install -e ".[dev,docs]"

JAX should be installed with the correct accelerator support for your system
(CPU, CUDA, or ROCm). Consult the JAX installation guide for platform-specific
instructions.

The ``boozmn`` reader relies on the ``netCDF4`` Python package, which is listed
as a core dependency.
