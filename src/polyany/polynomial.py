from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from .exponents import get_quadratic_exponents


class Polynomial:
    """A multivariate polynomial class.

    Represents a multivariate polynomial in the form:

    P(X) = ∑ c_i * x_1^e_i1 * x_2^e_i2 * ... * x_n^e_in

    where `c_i` are the coefficients and `e_ji` are the exponents of each monomial.

    Parameters
    ----------
    exponents : ArrayLike
        A nested sequence or a NumPy 2D-array with shape (n_monomials, n_vars),
        where each row contains the exponents of one monomial.
        The order of variables is assumed to be increasing, i.e.,
        [x_1, x_2, ..., x_n].
    coefficients : ArrayLike
        A sequence or a NumPy 1D-array with shape (n_monomials,). Containing the
        corresponding scalar multipliers of each monomial.

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
        A NumPy 1D-array with the corresponding coefficients.

    Raises
    ------
    TypeError
        - If the input exponents cannot be safely converted to a
        NumPy 2D-array of integers.
        - If the input coefficients cannot be safely converted to a
        NumPy 1D-array of floats.

    ValueError
        - If the number of exponents does not match the number of coefficients.
        - If the input arrays dimensions are inconsistent.
        - If the input exponents rows are not unique.
        - If any input exponent entry is negative.

    Notes
    -----
    The current implementation allows coefficients to be complex numbers,
    but complex polynomials are not yet officially supported and may produce
    unexpected behavior.

    Although attributes are publicly accessible, modifying them directly may lead
    to bugs and unexpected behavior.

    Examples
    --------
    >>> from polyany import Polynomial

    Create the polynomial: ``5*x_1**2*x_2*x_3**4*x_5 + 3*x_1*x_2 + 4*x_4**4*x_5**3``

    >>> exponents = [[1, 1, 0, 0, 0],
    ...              [0, 0, 0, 4, 3],
    ...              [2, 1, 4, 0, 1]]
    >>> coefficients = [3, 4, 5]
    >>> Polynomial(exponents, coefficients)
    3*x_1*x_2 + 4*x_4^4*x_5^3 + 5*x_1^2*x_2*x_3^4*x_5
    """

    def __init__(self, exponents: ArrayLike, coefficients: ArrayLike) -> None:
        input_exponents, input_coefficients = self._sanitize_inputs(
            exponents, coefficients
        )

        self.n_vars = input_exponents.shape[1]
        self.degree = np.max(np.sum(input_exponents, axis=1)).item()

        self.exponents, self.coefficients = self._sort_inputs(
            input_exponents, input_coefficients
        )

    def _sanitize_inputs(
        self, input_exponents: ArrayLike, input_coefficients: ArrayLike
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

    def _sort_inputs(
        self, exponents: np.ndarray, coefficients: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        monomials_degree = np.sum(exponents, axis=1)
        sorted_idx = np.lexsort((*exponents.T, monomials_degree))

        return exponents[sorted_idx], coefficients[sorted_idx]

    @classmethod
    def univariate(cls, coefficients: ArrayLike) -> Polynomial:
        """Creates a univariate polynomial from a coefficients vector

        This classmethod is a convenient shortcut to construct a univariate polynomial
        from a coefficients vector.

        Parameters
        ----------
        coefficients : ArrayLike
            The coefficients of the univariate polynomial, associated with increasing
            powers of the variable `x_1`.

        Returns
        -------
        Polynomial
            A univariate polynomial.

        Raises
        ------
        ValueError
            - If `coefficients` does not have exactly one dimension.

        Examples
        --------
        >>> Polynomial.univariate([1, 2, -3, -4, 5])
        1 + 2*x_1 - 3*x_1^2 - 4*x_1^3 + 5*x_1^4
        """
        converted_coefficients = np.asarray(coefficients)

        if converted_coefficients.ndim != 1:
            msg = (
                f"Coefficients must have 1 dimension, got {converted_coefficients.ndim}"
            )
            raise ValueError(msg)

        exponents = np.arange(0, len(converted_coefficients)).reshape(-1, 1)

        return cls(exponents, coefficients)

    @classmethod
    def quadratic_form(cls, matrix: ArrayLike) -> Polynomial:
        """Creates a quadratic form from its associated symmetric matrix

        Parameters
        ----------
        matrix : ArrayLike
            A nested sequence or a NumPy 2D array of shape (`n_vars`, `n_vars`) that
            representing the symmetric matrix associated with the quadratic form.

        Returns
        -------
        Polynomial
            A second-degree homogeneous multivariate polynomial, i.e, a quadratic form.

        Raises
        ------
        TypeError
            - If `matrix` is not safe-convertible to a
            NumPy 2D array with float entries.
        ValueError
            - If `matrix` does not have 2 dimensions.
            - If `matrix` is not square.

        Warns
        -----
        UserWarning
            - If `matrix`is not symmetric.

        Notes
        -----
        If `matrix` is not symmetric, its symmetric part is used instead,
        computed as `symmetric_part = (matrix + matrix.T) / 2`.

        Examples
        --------
        >>> matrix = [[5, 3, 2],
        ...           [3, 1, 0],
        ...           [2, 0, 7]]
        >>> Polynomial.quadratic_form(matrix)
        5*x_1^2 + 6*x_1*x_2 + x_2^2 + 4*x_1*x_3 + 7*x_3^2
        """
        try:
            converted_matrix = np.asarray(matrix).astype(
                dtype=np.float64, casting="safe", copy=True
            )
        except Exception as e:
            msg = (
                "Matrix must be safe-convertible to NumPy 2D-array with float entries."
            )
            raise TypeError(msg) from e

        if converted_matrix.ndim != 2:
            msg = f"Matrix must have 2 dimensions, got {converted_matrix.ndim}"
            raise ValueError(msg)

        if converted_matrix.shape[0] != converted_matrix.shape[1]:
            msg = f"Matrix must be square, got {converted_matrix.shape}"
            raise ValueError(msg)

        if not np.allclose(converted_matrix, converted_matrix.T):
            msg = "Matrix is not symmetric, its symmetric part will be considered"
            warnings.warn(msg, UserWarning, stacklevel=2)

            converted_matrix = (converted_matrix + converted_matrix.T) / 2

        n_vars = len(converted_matrix)
        index = np.arange(n_vars)

        upper_triangular_mask = index.reshape(-1, 1) < index
        converted_matrix[upper_triangular_mask] *= 2
        np.fill_diagonal(upper_triangular_mask, val=True)
        coefficients = converted_matrix[upper_triangular_mask]

        exponents = get_quadratic_exponents(n_vars)

        return cls(exponents, coefficients)

    def __call__(self, point: ArrayLike) -> np.float64:
        """Evaluate the polynomial at a given point

        Parameters
        ----------
        point : ArrayLike
            A point with `n_vars` components.

        Returns
        -------
        np.float64
            The result of evaluating the polynomial at `point`.

        Raises
        ------
        TypeError
            - If `point` cannot be safely converted to a NumPy 1D-array of floats.
        ValueError
            - If `point` does not have exactly one dimension.
            - If `point` does not have `n_vars` components.

        Examples
        --------
        For univariate polynomials:

        >>> poly = Polynomial.univariate([1, 2, 3])
        >>> poly([0])
        np.float64(1.0)
        >>> poly([2])
        np.float64(17.0)

        For multivariate polynomials:
        >>> exponents = [[0, 0],
        ...              [1, 0],
        ...              [0, 1],
        ...              [1, 1]]
        >>> coefficients = [9, 7, 5, 3]
        >>> poly = Polynomial(exponents, coefficients)
        >>> poly([0, 0])
        np.float64(9.0)
        >>> poly([1, 2])
        np.float64(32.0)
        """
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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.degree != other.degree or self.n_vars != other.n_vars:
            return False

        return np.allclose(self.coefficients, other.coefficients)

    def __lt__(self, other: object) -> bool:
        return NotImplemented

    def __le__(self, other: object) -> bool:
        return NotImplemented

    def __gt__(self, other: object) -> bool:
        return NotImplemented

    def __ge__(self, other: object) -> bool:
        return NotImplemented

    def shift(self, k: int = 1) -> Polynomial:
        """Shifts the polynomial variables.

        This method returns a new polynomial with its variables shifted.
        A positive shift adds extra variables (increasing all variable indices).
        A negative shift removes variables, but only if they are empty.


        Parameters
        ----------
        k : int, optional
            The shift count. If positive, adds `k` extra variables
            (increase the variable indices). If negative, remove the first `abs(k)`
            variables, but only if they are empty (all corresponding exponents
            are zero).

        Returns
        -------
        Polynomial
            A new polynomial with shifted variables.

        Raises
        ------
        TypeError
            - If `k` is not an int.
        ValueError
            - If `k` is negative and the number of variables after shifting
            would be less than one.
            - If any of the first `abs(k)` variables are
            not empty.

        Notes
        -----
        If `k` = 0 a copy of the polynomial is returned.

        The Python shift operators can be used as a syntactic sugar for this method.
        `poly >> 3` is equivalent to `poly.shift(3)`, and `poly << 2` is equivalent to
        `poly.shift(-2)`.

        This method is reversible as long as both directions are valid.

        - The statement `poly.shift(k).shift(-k)` will return a polynomial equal to the
        original object `poly`.

        - Likewise, if `poly.shift(-k)` is possible, then applying `shift(k)` after it
        will also return a copy of `poly`.

        Examples
        --------
        Adding extra variables (shift right), increases the variable indices.

        >>> poly = Polynomial.univariate([1, 2, 3])
        >>> poly
        1 + 2*x_1 + 3*x_1^2
        >>> poly.shift(2)
        1 + 2*x_3 + 3*x_3^2
        >>> poly >> 2 # equivalent syntax
        1 + 2*x_3 + 3*x_3^2

        Removing empty variables (shift left), decreases the variable indices.

        >>> poly = Polynomial([[0, 1], [0, 3], [0, 5]], [10, 20, 30])
        >>> poly
        10*x_2 + 20*x_2^3 + 30*x_2^5
        >>> poly.shift(-1)
        10*x_1 + 20*x_1^3 + 30*x_1^5
        >>> poly << 1 # equivalent syntax
        10*x_1 + 20*x_1^3 + 30*x_1^5
        """
        if not isinstance(k, int):
            msg = f"k must be an int, got {type(k)}."
            raise TypeError(msg)

        exponents = self.exponents.copy()
        coefficients = self.coefficients.copy()

        if k < 0:
            vars_to_remove = ", ".join(["x_" + str(idx + 1) for idx in range(abs(k))])

            if self.n_vars + k < 1:
                msg = (
                    f"Cannot remove ({vars_to_remove}), "
                    "at least one variable must remain, "
                    f"ensure that k >= {-self.n_vars + 1}."
                )
                raise ValueError(msg)

            rows_with_nonzero_exponents = np.any(exponents[:, : abs(k)] != 0, axis=1)
            has_nonzero_coefficients = bool(
                np.any(coefficients[rows_with_nonzero_exponents] != 0)
            )

            if has_nonzero_coefficients:
                msg = (
                    f"Cannot remove ({vars_to_remove}), "
                    "at least one associated coefficient is not zero."
                )
                raise ValueError(msg)

            exponents = exponents[~rows_with_nonzero_exponents, abs(k) :]
            coefficients = coefficients[~rows_with_nonzero_exponents]

        if k > 0:
            exponents = np.hstack(
                (
                    np.zeros(shape=(len(exponents), k), dtype=np.int_),
                    exponents,
                )
            )

        return self.__class__(exponents, coefficients)

    def __rshift__(self, other: int) -> Polynomial:
        """Adds extra variables to the Polynomial.

        A shorthand for `Polynomial.shift(k)` with `k > 0` using the right shift
        operator (`>>`). For more details, see the
        [`Polynomial.shift()`][polyany.polynomial.Polynomial.shift] method.

        Parameters
        ----------
        other : int
            The shift count. Must be a non-negative integer.

        Returns
        -------
        Polynomial
            A new polynomial with shifted variables.

        Raises
        ------
        ValueError
            - If the shift count (`other`) is negative.
        """
        if not isinstance(other, int):  # pragma: no cover
            return NotImplemented

        if other < 0:
            msg = "Shift count must be non-negative."
            raise ValueError(msg)

        return self.shift(other)

    def __lshift__(self, other: int) -> Polynomial:
        """Removes empty variables of the Polynomial.

        A shorthand for `Polynomial.shift(k)` with `k < 0` using the left shift
        operator (`<<`). For more details, see the
        [`Polynomial.shift()`][polyany.polynomial.Polynomial.shift] method.

        Parameters
        ----------
        other : int
            The shift count. Must be a non-negative integer.

        Returns
        -------
        Polynomial
            A new polynomial with shifted variables.

        Raises
        ------
        ValueError
            - If the shift count (`other`) is negative.
        """
        if not isinstance(other, int):  # pragma: no cover
            return NotImplemented

        if other < 0:
            msg = "Shift count must be non-negative."
            raise ValueError(msg)

        return self.shift(-other)
