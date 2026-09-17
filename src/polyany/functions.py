from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .matrix import MatrixPolynomial

if TYPE_CHECKING:
    from collections.abc import Sequence


def concatenate(
    polynomials: Sequence[MatrixPolynomial], axis: int = 0
) -> MatrixPolynomial:
    """Concatenate a sequence of matrix polynomials

    The coefficient matrices of the polynomials are concatenated
    (vertically or horizontally) with respect to each monomial.

    Parameters
    ----------
    polynomials : Sequence[MatrixPolynomial]
        A sequence (list or tuple) of matrix polynomials to concatenate.
    axis : int
        The axis along which the polynomials will be concatenated. Use 0 for vertical
        concatenation or 1 for horizontal concatenation. Default is 0.

    Returns
    -------
    concatenated_polynomial : MatrixPolynomial
        A new matrix polynomial with concatenated coefficients.

    Raises
    ------
    TypeError
        - If any of the sequence elements are not a MatrixPolynomial.
        - If axis is not an int.

    ValueError
        - If the input sequence is empty.
        - If axis is not 0 or 1.
        - If the polynomials shapes are inconsistent. If axis is 0 the
        polynomial shapes must match at dimension 1 (columns). If axis is 1 the
        polynomial shapes must match at dimension 0 (rows).

    Notes
    -----
    If a monomial exists in one polynomial but not in the others, a zeros matrix
    of appropriate shape is utilized.

    Examples
    --------
    >>> mpoly1 = MatrixPolynomial([[1], [2]], [np.eye(2), np.tri(2)])
    >>> mpoly2 = MatrixPolynomial([[1]], [np.ones((2,2))])
    >>> concatenate([mpoly1, mpoly2]) # defaults to vertical concatenation
    [[1. 0.]        [[1. 0.]
     [0. 1.]         [1. 1.]
     [1. 1.]         [0. 0.]
     [1. 1.]]*x_1 +  [0. 0.]]*x_1^2
    >>> concatenate([mpoly1, mpoly2], axis=1) # horizontal concatenation
    [[1. 0. 1. 1.]        [[1. 0. 0. 0.]
     [0. 1. 1. 1.]]*x_1 +  [1. 1. 0. 0.]]*x_1^2
    """
    if not all(isinstance(element, MatrixPolynomial) for element in polynomials):
        msg = "All elements to be concatenated must be MatrixPolynomial objects."
        raise TypeError(msg)

    if not isinstance(axis, int):
        msg = f"Axis must be an integer, got {type(axis).__name__}."
        raise TypeError(msg)

    if not polynomials:
        msg = "Input list/tuple must be non-empty."
        raise ValueError(msg)

    if axis not in (0, 1):
        msg = (
            "Axis must be 0 (vertical concatenation) or 1 (horizontal concatenation), "
            f"got {axis}."
        )
        raise ValueError(msg)

    match_axis = 1 - axis
    if not all(
        poly.shape[match_axis] == polynomials[0].shape[match_axis]
        for poly in polynomials
    ):
        msg = (
            f"To concatenate polynomials in the axis {axis}, the shapes musth match at "
            f"the dimension {match_axis}."
        )
        raise ValueError(msg)

    max_n_vars = max(poly.n_vars for poly in polynomials)

    stacked_exponents = np.vstack(
        [poly._domain_expansion(max_n_vars) for poly in polynomials]
    )
    sorted_idx = np.lexsort(stacked_exponents.T)
    stacked_exponents = stacked_exponents[sorted_idx]

    unique_mask = np.concatenate(
        ([True], (stacked_exponents[:-1] != stacked_exponents[1:]).any(axis=1))
    )
    unique_exponents = stacked_exponents[unique_mask]
    n_exps = len(unique_exponents)

    uniform_polynomials = []
    for poly in polynomials:
        dummy_poly = MatrixPolynomial._from_trusted_data(
            unique_exponents, np.zeros((n_exps, *poly.shape)), max_n_vars
        )
        uniform_polynomials.append(poly + dummy_poly)

    internal_axis = axis + 1
    concatenated_coefficients = np.concatenate(
        [poly.coefficients for poly in uniform_polynomials], axis=internal_axis
    )

    return MatrixPolynomial(unique_exponents, concatenated_coefficients)


def block(polynomials: Sequence[Sequence[MatrixPolynomial]]) -> MatrixPolynomial:
    """Create a matrix polynomial block

    A polynomial block is the structure formed by concatenating polynomials both
    vertically and horizontally.

    Parameters
    ----------
    polynomials : Sequence[Sequence[MatrixPolynomial]]
        A nested sequence (lists or tuples) of matrix polynomials to assemble.

    Returns
    -------
    block_polynomial : MatrixPolynomial
        A new matrix polynomial with assembled coefficients.

    Raises
    ------
    TypeError
        - If any of the inner elements are not a sequence (list or tuple).
        - If any of the inner sequence elements are not a MatrixPolynomial.

    Notes
    -----
    Similar to NumPy behavior, the inner sequence is concatenated horizontally and
    then the resulting polynomials are concatenated vertically.

    Examples
    --------
    >>> mpoly1 = MatrixPolynomial([[1], [2]], [np.eye(2), np.tri(2)])
    >>> mpoly2 = MatrixPolynomial([[1]], [np.ones((2,2))])
    >>> mpoly3 = MatrixPolynomial([[1]], [3*np.ones((3,2))])
    >>> mpoly4 = MatrixPolynomial([[2]], [np.arange(6).reshape(3, 2)])
    >>> block([[mpoly1, mpoly2], [mpoly3, mpoly4]])
    [[1. 0. 1. 1.]        [[1. 0. 0. 0.]
     [0. 1. 1. 1.]         [1. 1. 0. 0.]
     [3. 3. 0. 0.]         [0. 0. 0. 1.]
     [3. 3. 0. 0.]         [0. 0. 2. 3.]
     [3. 3. 0. 0.]]*x_1 +  [0. 0. 4. 5.]]*x_1^2
    """
    if not all(
        isinstance(inner_element, (list, tuple)) for inner_element in polynomials
    ):
        msg = "The inner elements must be a list/tuple of MatrixPolynomial objects."
        raise TypeError(msg)

    if not all(
        isinstance(element, MatrixPolynomial)
        for inner_list in polynomials
        for element in inner_list
    ):
        msg = "All elements to be assembled must be MatrixPolynomial objects."
        raise TypeError(msg)

    concatenated_rows = [concatenate(row, axis=1) for row in polynomials]

    return concatenate(concatenated_rows)
