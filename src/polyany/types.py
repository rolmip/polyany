from typing import TypeAlias

import numpy as np

from .polynomial import Polynomial

Scalar: TypeAlias = int | float | np.integer | np.floating
"""A numeric scalar that can be a builtin numeric type or a NumPy scalar."""
Algebraic: TypeAlias = Scalar | Polynomial
"""An algebraic element that can be a scalar or a Polynomial."""
