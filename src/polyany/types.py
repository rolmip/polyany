from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike

from .matrix import MatrixPolynomial
from .polynomial import Polynomial

Scalar: TypeAlias = int | float | np.integer | np.floating
"""A numeric scalar that can be a builtin numeric type or a NumPy scalar."""
ScalarAlgebraic: TypeAlias = Scalar | Polynomial
"""An algebraic element that can be a scalar or a scalar Polynomial."""
MatrixAlgebraic: TypeAlias = ArrayLike | MatrixPolynomial
"""An algebraic element that can be a matrix or a matrix Polynomial."""
