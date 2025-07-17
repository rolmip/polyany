from __future__ import annotations

from itertools import product

import numpy as np
import numpy.typing as npt


class Polynomial:
    def __init__(self, exponents: npt.ArrayLike, coefficients: npt.ArrayLike) -> None:
        input_exponents, input_coefficients = self._sanitize_inputs(
            exponents, coefficients
        )

        self.n_vars = input_exponents.shape[1]
        self.degree = np.max(np.sum(input_exponents, axis=1))

        self.exponents, self.coefficients = self._full_representation(
            input_exponents, input_coefficients
        )

    def _sanitize_inputs(
        self, input_exponents: npt.ArrayLike, input_coefficients: npt.ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        try:
            converted_coefficients = np.asarray(input_coefficients).astype(
                dtype=np.float64, casting="safe"
            )
        except Exception as e:
            msg = (
                "Coefficients must be safe-convertible to NumPy 1D-arrays "
                "with float entries."
            )
            raise TypeError(msg) from e

        try:
            converted_exponents = np.asarray(input_exponents).astype(
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

        if len(np.unique(converted_exponents, axis=0)) != len(converted_exponents):
            msg = "Exponents entries must be unique."
            raise ValueError(msg)

        if len(converted_exponents) != len(converted_coefficients):
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

        return converted_exponents, converted_coefficients

    def __repr__(self) -> str:
        # TODO(@ximiraxelo): truncate output for large polynomials

        monomials: list[str] = []
        for exponent, coefficient in zip(
            self.exponents, self.coefficients, strict=True
        ):
            if coefficient == 0:
                continue

            variables = "*".join(
                [
                    f"x_{idx + 1}^{deg}" if deg > 1 else f"x_{idx + 1}"
                    for idx, deg in enumerate(exponent)
                    if deg > 0
                ]
            )

            if float(coefficient).is_integer():
                coef_value = abs(int(coefficient))
            else:
                coef_value = abs(coefficient)

            coef_str = "" if coef_value == 1 and variables else str(coef_value)

            term = f"{coef_str}{'*' if variables and coef_str else ''}{variables}"
            sign = " - " if coefficient < 0 else (" + " if monomials else "")

            monomials.append(f"{sign}{term}" if sign else term)

        if not monomials:
            return "0"

        monomials[0] = monomials[0].replace(" ", "")

        return "".join(monomials)

    def _full_representation(
        self, exponents: np.ndarray, coefficients: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n_monomials = len(exponents)

        input_monomials = {
            tuple(exponents[i]): coefficients[i] for i in range(n_monomials)
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

        monomials_degree = np.sum(full_exponents, axis=1)
        sorted_idx = np.lexsort((*full_exponents.T, monomials_degree))
        full_exponents = full_exponents[sorted_idx]
        full_coefficients = full_coefficients[sorted_idx]

        return full_exponents, full_coefficients

    @classmethod
    def univariate(cls, coefficients: npt.ArrayLike) -> Polynomial:
        converted_coefficients = np.asarray(coefficients)

        if converted_coefficients.ndim != 1:
            msg = (
                f"Coefficients must have 1 dimension, got {converted_coefficients.ndim}"
            )
            raise ValueError(msg)

        exponents = np.arange(0, len(converted_coefficients)).reshape(-1, 1)

        return cls(exponents, coefficients)

    def __call__(self, point: npt.ArrayLike) -> np.float64:
        try:
            converted_point = np.asarray(point).astype(dtype=np.float64, casting="safe")
        except Exception as e:
            msg = (
                "Point must be safe-convertible to NumPy 1D-arrays with float entries."
            )
            raise TypeError(msg) from e

        if converted_point.ndim != 1:
            msg = f"Point must have 1 dimension, got {converted_point.ndim}."
            raise ValueError(msg)

        if len(converted_point) != self.n_vars:
            msg = (
                f"Point must have {self.n_vars} component(s), "
                f"got {len(converted_point)}."
            )
            raise ValueError(msg)

        if np.all(converted_point == 0):
            return self.coefficients[0]

        return self.coefficients @ np.prod(
            np.power(converted_point, self.exponents), axis=1
        )
