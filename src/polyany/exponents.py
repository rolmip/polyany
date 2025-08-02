import numpy as np


def get_quadratic_exponents(n_vars: int) -> np.ndarray:
    eye = np.eye(n_vars, dtype=np.int16)
    i, j = np.triu_indices(n_vars)

    return eye[i] + eye[j]


def domain_expansion(exponents: np.ndarray, expanded_n_vars: int) -> np.ndarray:
    exponents = exponents.copy()
    extra_vars = expanded_n_vars - exponents.shape[1]

    if extra_vars > 0:
        exponents = np.hstack(
            (
                exponents,
                np.zeros(shape=(len(exponents), extra_vars), dtype=np.int16),
            )
        )

    return exponents
