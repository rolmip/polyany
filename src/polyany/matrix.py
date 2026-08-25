from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from polyany.types import MatrixAlgebraic

from .base import BasePolynomial


class MatrixPolynomial(BasePolynomial):
    """A matrix multivariate polynomial class.

    Represents a multivariate polynomial in the form:

    P(X) = ∑ C_i * x_1^e_i1 * x_2^e_i2 * ... * x_n^e_in

    where `C_i` are the matrix coefficients and `e_ji` are the exponents of each
    monomial.

    Parameters
    ----------
    exponents : ArrayLike
        A nested sequence or a NumPy 2D-array with shape (n_monomials, n_vars),
        where each row contains the exponents of one monomial.
        The order of variables is assumed to be increasing, i.e.,
        [x_1, x_2, ..., x_n].
    coefficients : ArrayLike
        A sequence or a NumPy 3D-array with shape (n_monomials, n_rows, n_cols).
        Containing the corresponding matrix multipliers of each monomial.

    Attributes
    ----------
    n_vars : int
        Number of variables in the polynomial.
    degree : int
        Total degree of the polynomial.
    shape  : tuple of ints
        Common shape of the matrices (n_rows, n_cols).
    exponents : np.ndarray
        A NumPy 2D-array representing the exponents
        of the polynomial.
    coefficients : np.ndarray
        A NumPy 3D-array with the corresponding matrix coefficients.

    Raises
    ------
    TypeError
        - If the input exponents cannot be safely converted to a
        NumPy 2D-array of integers.
        - If the input coefficients cannot be safely converted to a
        NumPy 3D-array of floats.

    ValueError
        - If the number of exponents does not match the number of coefficients.
        - If the input arrays dimensions are inconsistent.
        - If the input exponents rows are not unique.
        - If any input exponent entry is negative.

    Notes
    -----
    The current implementation allows matrices to have complex entries,
    but complex polynomials are not yet officially supported and may produce
    unexpected behavior.

    Although 1x1 matrices are allowed, if you intend to create a polynomial with
    scalar coefficients, check the [`Polynomial`][polyany.polynomial.Polynomial] class
    for more efficient operations and manipulations.

    Although attributes are publicly accessible, modifying them directly may lead
    to bugs and unexpected behavior.

    Examples
    --------
    >>> from polyany import MatrixPolynomial

    Create the matrix polynomial:

    ```
    [[1, 0]       [[0, 1]
     [0, 1]]*x_1 +  [2, 3]]*x_2
    ```

    >>> exponents = [
    ...    [1, 0],
    ...    [0, 1],
    ... ]
    >>> C_1 = np.eye(2)
    >>> C_2 = np.arange(4).reshape(2, 2)
    >>> coefficients = [C_1, C_2]
    >>> MatrixPolynomial(exponents, coefficients)
    [[1. 0.]        [[0. 1.]
     [0. 1.]]*x_1 +  [2. 3.]]*x_2
    """

    __array_ufunc__ = None

    def __init__(self, exponents: ArrayLike, coefficients: ArrayLike) -> None:
        super().__init__(exponents, coefficients)

        self.shape = self.coefficients.shape[1:]

    def _sanitize_coefficients(self, coefficients: ArrayLike) -> np.ndarray:
        try:
            converted_coefficients = np.asarray(coefficients).astype(
                dtype=np.float64, casting="safe"
            )
        except Exception as e:
            msg = (
                "Matrix coefficients must be safe-convertible to NumPy 3D-arrays "
                "with float entries."
            )
            raise TypeError(msg) from e

        if converted_coefficients.ndim != 3:
            msg = (
                "Matrix coefficients must have 3 dimensions, "
                f"got {converted_coefficients.ndim}."
            )
            raise ValueError(msg)

        if converted_coefficients.size == 0:
            msg = "Coefficients must have at least one element, got 0."
            raise ValueError(msg)

        return converted_coefficients

    def __repr__(self) -> str:
        formatted_monomials: list[list[str]] = []
        with np.printoptions(
            linewidth=1_000,
            threshold=50,
        ):
            for exponent, coefficient in zip(
                self.exponents, self.coefficients, strict=True
            ):
                if np.all(coefficient == 0):
                    continue

                variables = "*".join(
                    [
                        f"x_{idx + 1}^{deg}" if deg > 1 else f"x_{idx + 1}"
                        for idx, deg in enumerate(exponent)
                        if deg > 0
                    ]
                )

                monomial_str = f"{coefficient}{'*' if variables else ''}{variables}"

                split_monomial = monomial_str.split("\n")
                max_size = len(split_monomial[-1])
                formatted_monomials.append(
                    [line.ljust(max_size) for line in split_monomial]
                )

            if not formatted_monomials:
                return str(np.zeros(self.shape))

        formatted_lines: list[str] = []
        last_line = len(formatted_monomials[0]) - 1
        for idx, line in enumerate(zip(*formatted_monomials, strict=True)):
            separator = " + " if idx == last_line else "   "
            formatted_lines.append(separator.join(line))

        return "\n".join(formatted_lines)

    def __add__(self, other: MatrixAlgebraic) -> MatrixPolynomial:
        """Addition with another matrix polynomial, matrix or scalar

        Parameters
        ----------
        other : MatrixAlgebraic
            The operand in the addition.
            A scalar can be an int, float, or NumPy scalars.
            A matrix can be a NumPy 2D-array, nested lists or nested tuples.

        Returns
        -------
        MatrixPolynomial
            A new matrix polynomial representing the sum.
        """
        if not isinstance(other, ALGEBRAIC_TYPE):  # pragma: no cover
            return NotImplemented

        if isinstance(other, SCALAR_TYPE):
            broadcasted = np.broadcast_to(other, self.shape)
            return self._add_matrix(broadcasted)

        if isinstance(other, MATRIX_TYPE):
            return self._add_matrix(other)

        return self._add_polynomial(other)

    def _add_matrix(self, other: ArrayLike) -> MatrixPolynomial:
        try:
            other = np.asarray(other).astype(
                self.coefficients.dtype, casting="safe", copy=False
            )
        except Exception as e:
            msg = (
                "Operand must be safe-convertible to NumPy 2D-arrays "
                "with float entries."
            )
            raise TypeError(msg) from e

        if other.shape != self.shape:
            msg = (
                f"Cannot add matrix of shape {other.shape} to "
                f"a polynomial of shape {self.shape}"
            )
            raise ValueError(msg)

        coefficients = self.coefficients.copy()
        exponents = self.exponents.copy()

        has_constant_term = (self.exponents[0] == 0).all()

        if has_constant_term:
            coefficients[0] += other
        else:
            exponents = np.vstack(
                (np.zeros((1, self.n_vars), dtype=exponents.dtype), exponents)
            )
            coefficients = np.concatenate((np.expand_dims(other, 0), coefficients))

        return self.__class__(exponents, coefficients)

    def _add_polynomial(self, other: MatrixPolynomial) -> MatrixPolynomial:
        if other.shape != self.shape:
            msg = (
                f"Cannot add polynomial of shape {other.shape} to "
                f"a polynomial of shape {self.shape}"
            )
            raise ValueError(msg)

        max_n_vars = max(self.n_vars, other.n_vars)

        if self.n_vars < max_n_vars:
            self._domain_expansion(max_n_vars)
        else:
            other._domain_expansion(max_n_vars)

        stacked_exponents = np.vstack((self.exponents, other.exponents))
        stacked_coefficients = np.concatenate((self.coefficients, other.coefficients))

        sorted_idx = np.lexsort(stacked_exponents.T)
        coefficients = stacked_coefficients[sorted_idx]
        exponents = stacked_exponents[sorted_idx]

        changes = (exponents[1:] != exponents[:-1]).any(axis=1)
        boundaries = np.concatenate(([0], np.nonzero(changes)[0] + 1))

        unique_exponents = exponents[boundaries]
        unique_coefficients = np.add.reduceat(coefficients, boundaries)

        return self.__class__(unique_exponents, unique_coefficients)

    def __sub__(self, other: MatrixAlgebraic) -> MatrixPolynomial:
        """Subtraction with another matrix polynomial, matrix or scalar

        Parameters
        ----------
        other : MatrixAlgebraic
            The operand in the subtraction.
            A scalar can be an int, float, or NumPy scalars.
            A matrix can be a NumPy 2D-array, nested lists or nested tuples.

        Returns
        -------
        MatrixPolynomial
            A new matrix polynomial representing the difference.
        """
        if isinstance(other, (SCALAR_TYPE, MatrixPolynomial)):
            return self.__add__(-other)

        try:
            other = np.negative(other)
        except Exception as e:
            msg = "Operand must be safe-convertible to NumPy array"
            raise TypeError(msg) from e

        return self.__add__(other)

    def __radd__(self, other: MatrixAlgebraic) -> MatrixPolynomial:
        return self.__add__(other)

    def __rsub__(self, other: MatrixAlgebraic) -> MatrixPolynomial:
        return (-self).__add__(other)


SCALAR_TYPE = (int, float, np.integer, np.floating)
MATRIX_TYPE = (list, tuple, np.ndarray)
ALGEBRAIC_TYPE = (*SCALAR_TYPE, *MATRIX_TYPE, MatrixPolynomial)
