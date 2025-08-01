import numpy as np


def get_quadratic_exponents(n_vars: int) -> np.ndarray:
    eye = np.eye(n_vars, dtype=np.int16)
    i, j = np.triu_indices(n_vars)

    return eye[i] + eye[j]
