import numpy as np
from numpy.typing import ArrayLike

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

        return converted_coefficients

    def __repr__(self) -> str:
        formatted_monomials: list[list[str]] = []
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

            with np.printoptions(
                linewidth=1_000,
                threshold=50,
            ):
                monomial_str = f"{coefficient}{'*' if variables else ''}{variables}"

            split_monomial = monomial_str.split("\n")
            max_size = len(split_monomial[-1])
            formatted_monomials.append(
                [line.ljust(max_size) for line in split_monomial]
            )

        formatted_lines: list[str] = []
        last_line = len(formatted_monomials[0]) - 1
        for idx, line in enumerate(zip(*formatted_monomials, strict=True)):
            separator = " + " if idx == last_line else "   "
            formatted_lines.append(separator.join(line))

        return "\n".join(formatted_lines)
