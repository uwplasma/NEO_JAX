import os
import sys

import jax

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Match Fortran double precision behavior in tests.
jax.config.update("jax_enable_x64", True)
