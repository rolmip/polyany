from itertools import product

import numpy as np
import numpy.typing as npt


class Polynomial:
    def __init__(self, exponents: npt.ArrayLike, coefficients: npt.ArrayLike) -> None:
        self.input_exponents, self.input_coefficients, self.degree = (
            self._sanitize_inputs(exponents, coefficients)
        )
        self.n_monomials, self.n_vars = self.input_exponents.shape

        self.exponents, self.coefficients = self._full_representation()

    def _sanitize_inputs(
        self, input_exponents: npt.ArrayLike, input_coefficients: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray, np.int_]:
        try:
            converted_coefficients = (
                np.array(input_coefficients)
                .astype(dtype=np.float64, casting="safe")
                .squeeze()
            )
        except Exception as e:
            msg = (
                "Coefficients must be safe-convertible to NumPy 1D-arrays "
                "with float entries."
            )
            raise TypeError(msg) from e

        try:
            converted_exponents = np.array(input_exponents).astype(
                dtype=np.int_, casting="safe"
            )
        except Exception as e:
            msg = (
                "Exponents must be safe-convertible to NumPy 2D-arrays "
                "with int entries."
            )
            raise TypeError(msg) from e

        if converted_exponents.ndim != 2:
            msg = f"Exponents must have 2 dimensions, got {converted_exponents.ndim}."
            raise ValueError(msg)

        if converted_coefficients.ndim != 1:
            msg = (
                "Coefficients must have 1 dimension, "
                f"got {converted_coefficients.ndim}."
            )
            raise ValueError(msg)

        if converted_exponents.shape[0] != converted_coefficients.shape[0]:
            msg = (
                "Number of exponents and coefficients must match, "
                f"got {converted_exponents.shape[0]} exponents / "
                f"{converted_coefficients.shape[0]} coefficients."
            )
            raise ValueError(msg)

        if not np.all(converted_exponents >= 0):
            msg = (
                "PolyAny is not yet able to handle nonlinear polynomials. "
                "Make sure that all exponents are >= 0."
            )
            raise ValueError(msg)

        monomials_degree = np.sum(converted_exponents, axis=1)
        sorted_idx = np.lexsort((*converted_exponents.T, monomials_degree))
        converted_coefficients = converted_coefficients[sorted_idx]
        converted_exponents = converted_exponents[sorted_idx]

        polynomial_degree = np.max(monomials_degree)

        return converted_exponents, converted_coefficients, polynomial_degree

    def __repr__(self) -> str:
        # TODO(@ximiraxelo): truncate output for large polynomials

        monomials: list[str] = []
        for exponent, coefficient in zip(
            self.input_exponents, self.input_coefficients, strict=True
        ):
            variables = "*".join(
                [
                    f"x_{idx + 1}^{deg}" if deg > 1 else f"x_{idx + 1}"
                    for idx, deg in enumerate(exponent)
                    if deg > 0
                ]
            )

            monomials.append(f"{' + ' if coefficient >= 0 else ' - '}")

            if float(coefficient).is_integer():
                coef_value = abs(int(coefficient))
            else:
                coef_value = abs(coefficient)

            monomials.append(f"{coef_value}{'*' if variables else ''}{variables}")

        if self.input_coefficients[0] >= 0:
            monomials.pop(0)

        return "".join(monomials)

    def _full_representation(self) -> tuple[np.ndarray, np.ndarray]:
        input_monomials = {
            tuple(self.input_exponents[i]): self.input_coefficients[i]
            for i in range(self.n_monomials)
        }

        full_exponents = np.array(
            [
                exponent
                for exponent in product(range(self.degree + 1), repeat=self.n_vars)
                if sum(exponent) <= self.degree
            ]
        )

        full_coefficients = np.array(
            [input_monomials.get(tuple(exponent), 0.0) for exponent in full_exponents]
        )

        return full_exponents, full_coefficients
