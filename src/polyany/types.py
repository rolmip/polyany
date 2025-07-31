from typing import TypeAlias

import numpy as np

from .polynomial import Polynomial

Scalar: TypeAlias = int | float | np.integer | np.floating
Algebraic: TypeAlias = Scalar | Polynomial
