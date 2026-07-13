from abc import ABC, abstractmethod
from typing import TypeVar

import numpy as np
from numpy.typing import ArrayLike

TBasePolynomial = TypeVar("TBasePolynomial", bound="BasePolynomial")


class BasePolynomial(ABC):
    """A multivariate polynomial abstract base class.

    Abstract implementation of the core structure of polynomials.

    Parameters
    ----------
    exponents : ArrayLike
        A nested sequence or a NumPy 2D-array with shape (n_monomials, n_vars),
        where each row contains the exponents of one monomial.
        The order of variables is assumed to be increasing, i.e.,
        [x_1, x_2, ..., x_n].
    coefficients : ArrayLike
        A sequence or NumPy array of coefficients corresponding to each monomial.
        The exact shape is defined by the concrete subclasses.

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
        A NumPy array of coefficients with shape defined by the concrete subclass.

    Raises
    ------
    TypeError
        - If the input exponents cannot be safely converted to a
        NumPy 2D-array of integers.

    ValueError
        - If the number of exponents does not match the number of coefficients.
        - If the input arrays dimensions are inconsistent.
        - If the input exponents rows are not unique.
        - If any input exponent entry is negative.

    Notes
    -----
    Although attributes are publicly accessible, modifying them directly may lead
    to bugs and unexpected behavior.
    """

    def __init__(self, exponents: ArrayLike, coefficients: ArrayLike) -> None:
        self.exponents = self._sanitize_exponents(exponents)
        self.coefficients = self._sanitize_coefficients(coefficients)

        self._validate_inputs()
        self._sort_and_check_inputs()

        self.n_vars = self.exponents.shape[1]
        self.degree = np.max(np.sum(self.exponents, axis=1)).item()

    def _sanitize_exponents(self, input_exponents: ArrayLike) -> np.ndarray:
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

        if not np.all(converted_exponents >= 0):
            msg = (
                "PolyAny is not yet able to handle nonlinear polynomials. "
                "Make sure that all exponents are >= 0."
            )
            raise ValueError(msg)

        return converted_exponents

    @abstractmethod
    def _sanitize_coefficients(
        self, coefficients: ArrayLike
    ) -> np.ndarray:  # pragma: no cover
        pass

    def _validate_inputs(self) -> None:
        if len(self.exponents) != len(self.coefficients):
            msg = (
                "Number of exponents and coefficients must match, "
                f"got {self.exponents.shape[0]} exponents / "
                f"{self.coefficients.shape[0]} coefficients."
            )
            raise ValueError(msg)

    def _sort_and_check_inputs(self) -> None:
        monomials_degree = np.sum(self.exponents, axis=1)
        sorted_idx = np.lexsort((*self.exponents.T, monomials_degree))

        self.exponents, self.coefficients = (
            self.exponents[sorted_idx],
            self.coefficients[sorted_idx],
        )

        if len(self.exponents) > 1 and np.any(
            np.all(self.exponents[1:] == self.exponents[:-1], axis=1)
        ):
            msg = "Exponents entries must be unique."
            raise ValueError(msg)

    def _domain_expansion(self, n_vars: int) -> None:
        extra_vars = n_vars - self.exponents.shape[1]

        if extra_vars > 0:
            self.exponents = np.hstack(
                (
                    self.exponents,
                    np.zeros(shape=(len(self.exponents), extra_vars), dtype=np.int16),
                )
            )

    @staticmethod
    def _get_quadratic_exponents(n_vars: int) -> np.ndarray:
        eye = np.eye(n_vars, dtype=np.int16)
        i, j = np.triu_indices(n_vars)

        return eye[i] + eye[j]

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

    def __neg__(self: TBasePolynomial) -> TBasePolynomial:
        """The negation of the polynomial.

        All coefficients are multiplied by `-1`. The exponents remain unchanged.

        Returns
        -------
        Self
            A new polynomial with negated coefficients.
        """
        return self.__class__(self.exponents.copy(), -self.coefficients)
