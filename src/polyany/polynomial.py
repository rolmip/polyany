from __future__ import annotations

import re
import warnings
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .exponents import domain_expansion, get_quadratic_exponents

if TYPE_CHECKING:  # pragma: no cover
    from .types import Algebraic, Scalar


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

    def _repr_latex_(self) -> str:
        """LaTeX representation of a polynomial

        Returns
        -------
        str
            The LaTeX expression enclosed by `$` signs.

        Notes
        -----
        This method is primarily used to produce rich outputs in Jupyter Notebooks.
        """
        representation = str(self).replace("*", r"\,")
        representation = re.sub(r"x_(\d+)", r"x_{\1}", representation)
        representation = re.sub(r"\^(\d+)", r"^{\1}", representation)

        return f"${representation}$"

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

    @classmethod
    def zeros(cls, n_vars: int) -> Polynomial:
        """Create a zero polynomial.

        Returns a polynomial with a single monomial (the constant 0)
        in `n_vars` variables.

        Parameters
        ----------
        n_vars : int
            Number of variables in the polynomial.

        Returns
        -------
        Polynomial
            A zero polynomial.

        Raises
        ------
        TypeError
            - If `n_vars` is not an int.
        ValueError
            - If `n_vars` is less than 1.

        Notes
        -----
        Primarily intended for internal use in specific cases.

        Examples
        --------
        >>> poly = Polynomial.zeros(3)
        >>> poly
        0
        >>> poly.exponents
        array([[0, 0, 0]])
        >>> poly.coefficients
        array([0.])
        """
        if not isinstance(n_vars, int):
            msg = f"n_vars must be an int, got {type(n_vars)}."
            raise TypeError(msg)

        if n_vars < 1:
            msg = f"n_vars must be greater or equal to 1, got {n_vars}"
            raise ValueError(msg)

        return cls(np.zeros((1, n_vars), dtype=np.int_), np.zeros(1))

    def prune(self) -> Polynomial:
        """Prune the empty monomials of a polynomial.

        Removes all monomials whose associated coefficients are exactly zero.

        Returns
        -------
        Polynomial
            A pruned polynomial, containing only monomials with non-zero coefficients.

        Notes
        -----
        If all coefficients are zero, a [`zeros`][polyany.Polynomial.zeros] polynomial
        with the same number of variables is returned.

        Examples
        --------
        >>> poly = Polynomial.univariate([1, 0, 0, 1])
        >>> poly.exponents
        array([[0],
               [1],
               [2],
               [3]])

        This polynomial has four terms, but only the first and last have a
        non-zero coefficient.

        >>> pruned = poly.prune()
        >>> pruned.exponents
        array([[0],
               [3]])

        The result keeps only the non-empty monomials, discarding all others.
        """
        non_empty_mask = self.coefficients != 0

        if not np.any(non_empty_mask):
            return self.__class__.zeros(self.n_vars)

        return self.__class__(
            self.exponents[non_empty_mask], self.coefficients[non_empty_mask]
        )

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
            if np.all(self.exponents[0] == 0):
                return self.coefficients[0]
            return np.float64(0)

        return self.coefficients @ np.prod(
            np.power(converted_point, self.exponents), axis=1
        )

    def __neg__(self) -> Polynomial:
        """The negation of the polynomial.

        All coefficients are multiplied by `-1`. The exponents remain unchanged.

        Returns
        -------
        Polynomial
            A new polynomial with negated coefficients.
        """
        return self.__class__(self.exponents.copy(), -self.coefficients)

    def __add__(self, other: object) -> Polynomial:
        """Addition with another polynomial or scalar

        Parameters
        ----------
        other : object
            The value to be added. A scalar can be an int, float, or NumPy scalars.

        Returns
        -------
        Polynomial
            A new polynomial representing the sum.
        """
        if not isinstance(other, ALGEBRAIC_TYPE):  # pragma: no cover
            return NotImplemented

        if isinstance(other, SCALAR_TYPE):
            return self._add_scalar(other)

        return self._add_polynomial(other)

    def _add_scalar(self, other: Scalar) -> Polynomial:
        coefficients = self.coefficients.copy()
        exponents = self.exponents.copy()

        has_constant_term = np.all(self.exponents[0] == 0)

        if has_constant_term:
            coefficients[0] += other
        else:
            exponents = np.vstack(
                (np.zeros((1, self.n_vars), dtype=exponents.dtype), exponents)
            )
            coefficients = np.concatenate((np.atleast_1d(other), coefficients))

        return self.__class__(exponents, coefficients)

    def _add_polynomial(self, other: Polynomial) -> Polynomial:
        max_n_vars = max(self.n_vars, other.n_vars)

        self_exponents = domain_expansion(self.exponents, max_n_vars)
        other_exponents = domain_expansion(other.exponents, max_n_vars)

        stacked_exponents = np.vstack((self_exponents, other_exponents))
        stacked_coefficients = np.concatenate((self.coefficients, other.coefficients))

        exponents, indices = np.unique(stacked_exponents, axis=0, return_inverse=True)
        coefficients = np.zeros(len(exponents))
        np.add.at(coefficients, indices, stacked_coefficients)

        return self.__class__(exponents, coefficients)

    def __sub__(self, other: Algebraic) -> Polynomial:
        """Subtraction with another polynomial or scalar

        Parameters
        ----------
        other : Algebraic
            The value to be subtracted. A scalar can be an int, float,
            or NumPy scalars.

        Returns
        -------
        Polynomial
            A new polynomial representing the difference.
        """
        return self.__add__(-other)

    def __radd__(self, other: Scalar) -> Polynomial:
        return self.__add__(other)

    def __rsub__(self, other: Scalar) -> Polynomial:
        return (-self).__add__(other)

    def __mul__(self, other: object) -> Polynomial:
        """Multiplication with another polynomial or scalar

        Parameters
        ----------
        other : object
            The value to be multiplied. A scalar can be an int, float, or NumPy scalars.

        Returns
        -------
        Polynomial
            A new polynomial representing the multiplication.
        """
        if not isinstance(other, ALGEBRAIC_TYPE):  # pragma: no cover
            return NotImplemented

        if isinstance(other, SCALAR_TYPE):
            return self._mul_scalar(other)

        return self._mul_polynomial(other)

    def _mul_scalar(self, other: Scalar) -> Polynomial:
        if other == 0:
            return self.__class__.zeros(self.n_vars)

        coefficients = self.coefficients * other

        return self.__class__(self.exponents.copy(), coefficients)

    def _mul_polynomial(self, other: Polynomial) -> Polynomial:
        max_n_vars = max(self.n_vars, other.n_vars)

        self_exponents = domain_expansion(self.exponents, max_n_vars)
        other_exponents = domain_expansion(other.exponents, max_n_vars)

        cross_exponents = (
            self_exponents[np.newaxis, :, :] + other_exponents[:, np.newaxis, :]
        ).reshape(-1, max_n_vars)

        cross_coefficients = (
            self.coefficients[np.newaxis, :] * other.coefficients[:, np.newaxis]
        ).ravel()

        exponents, indices = np.unique(cross_exponents, axis=0, return_inverse=True)
        coefficients = np.zeros(len(exponents))
        np.add.at(coefficients, indices, cross_coefficients)

        return self.__class__(exponents, coefficients)

    @np.errstate(divide="raise")
    def __truediv__(self, other: Scalar) -> Polynomial:
        """Division with a scalar

        Parameters
        ----------
        other : Scalar
            The value to divide the polynomial by.

        Returns
        -------
        Polynomial
            A new polynomial representing the division.

        Raises
        ------
        ZeroDivisionError
            - If `other` is a builtin scalar and equal to zero.
        FloatingPointError
            - If `other` is a NumPy scalar and equal to zero.

        Notes
        -----
        Currently, division can only be performed between polynomials and scalars.
        """
        if not isinstance(other, SCALAR_TYPE):  # pragma: no cover
            return NotImplemented

        return self.__mul__(1 / other)

    def __rmul__(self, other: Scalar) -> Polynomial:
        return self.__mul__(other)

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

    def partial(self, var_index: int) -> Polynomial:
        """Partial derivative of a polynomial

        Computes the partial derivative of the polynomial with respect to the variable
        indexed by `var_index`.

        Parameters
        ----------
        var_index : int
            The variable index to perform the partial derivative (zero-based).

        Returns
        -------
        Polynomial
            The resulting polynomial after differentiation.

        Raises
        ------
        TypeError
            - If `var_index` is not an int.
        ValueError
            - If `var_index` is outside the valid range [0, `n_vars` - 1].

        Examples
        --------
        >>> poly = Polynomial([[1, 0], [2, 1]], [3, 5])
        >>> poly
        3*x_1 + 5*x_1^2*x_2
        >>> poly.partial(0)
        3 + 10*x_1*x_2
        >>> poly.partial(1)
        5*x_1^2
        """
        if not isinstance(var_index, int):
            msg = f"var_index must be an int, got {type(var_index)}."
            raise TypeError(msg)

        if not (0 <= var_index < self.n_vars):
            if self.n_vars == 1:
                msg = "For a univariate polynomial, var_index must be 0"
            else:  # pragma: no cover
                msg = f"var_index must be between 0 and {self.n_vars - 1} (inclusive)"
            msg += f", got {var_index}."
            raise ValueError(msg)

        exponents = self.exponents.copy()
        coefficients = self.coefficients.copy()

        coefficients *= exponents[:, var_index]
        exponents[:, var_index] = np.maximum(0, exponents[:, var_index] - 1)

        non_empty_mask = coefficients != 0

        if not np.any(non_empty_mask):
            return self.__class__.zeros(self.n_vars)

        exponents = exponents[non_empty_mask]
        coefficients = coefficients[non_empty_mask]

        return self.__class__(exponents, coefficients)


SCALAR_TYPE = (int, float, np.integer, np.floating)
ALGEBRAIC_TYPE = (*SCALAR_TYPE, Polynomial)
