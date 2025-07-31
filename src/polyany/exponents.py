import math
from collections.abc import Generator

import numpy as np


def domain_expansion(
    exponents: np.ndarray, coefficients: np.ndarray, expanded_n_vars: int
) -> tuple[np.ndarray, np.ndarray]:
    exponents = exponents.copy()
    coefficients = coefficients.copy()
    extra_vars = expanded_n_vars - exponents.shape[1]

    if extra_vars > 0:
        exponents = np.hstack(
            (
                exponents,
                np.zeros(shape=(len(exponents), extra_vars), dtype=np.int16),
            )
        )

    return exponents, coefficients


def get_full_exponents(n_vars: int, degree: int) -> np.ndarray:
    count = math.comb(n_vars + degree, degree)
    dtype = np.int16 if n_vars == 1 else (np.int16, n_vars)

    return np.fromiter(
        _full_exponents_generator(n_vars, degree), dtype=dtype, count=count
    ).reshape(-1, n_vars)


def _full_exponents_generator(
    n_vars: int,
    degree: int,
    var_index: int = 0,
    current_exponents: list[int] | None = None,
    remaining_degree: int | None = None,
) -> Generator[tuple[int, ...] | int]:
    if current_exponents is None:
        current_exponents = [0] * n_vars
    if remaining_degree is None:
        remaining_degree = degree

    if var_index == n_vars:
        if n_vars == 1:
            yield current_exponents[0]
        else:
            yield tuple(current_exponents)
        return

    for current_degree in range(remaining_degree + 1):
        current_exponents[var_index] = current_degree
        yield from _full_exponents_generator(
            n_vars,
            degree,
            var_index + 1,
            current_exponents,
            remaining_degree - current_degree,
        )
