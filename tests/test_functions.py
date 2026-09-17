import numpy as np
import pytest

import polyany.functions as pa
from polyany import MatrixPolynomial


@pytest.mark.parametrize("axis", [0, 1])
def test_concatenate_square(axis):
    mpoly = MatrixPolynomial(
        [[0, 0], [1, 0], [0, 1]], [np.eye(2), np.tri(2), np.vander([1, 2])]
    )
    polynomials = [mpoly, mpoly, mpoly]

    concatenated = pa.concatenate(polynomials, axis=axis)
    concatenated_coefficients = np.concatenate(
        [polynomial.coefficients for polynomial in polynomials], axis=axis + 1
    )

    assert concatenated.n_vars == mpoly.n_vars
    assert concatenated.degree == mpoly.degree
    assert np.array_equal(concatenated.exponents, mpoly.exponents)
    assert np.array_equal(concatenated.coefficients, concatenated_coefficients)


@pytest.mark.parametrize("axis", [0, 1])
def test_concatenate_different_n_vars(axis):
    mpoly1 = MatrixPolynomial(
        [[0, 0], [1, 0], [0, 1]], [np.eye(2), np.tri(2), np.vander([1, 2])]
    )
    mpoly2 = MatrixPolynomial([[0]], [np.eye(2)])
    polynomials = [mpoly1, mpoly2]

    concatenated = pa.concatenate(polynomials, axis=axis)

    coefficient_0 = np.concatenate([np.eye(2), np.eye(2)], axis=axis)
    coefficient_1 = np.concatenate([np.tri(2), np.zeros((2, 2))], axis=axis)
    coefficient_2 = np.concatenate([np.vander([1, 2]), np.zeros((2, 2))], axis=axis)

    assert concatenated.n_vars == 2
    assert concatenated.degree == 1
    assert np.array_equal(concatenated.exponents, mpoly1.exponents)
    assert np.array_equal(concatenated.coefficients[0], coefficient_0)
    assert np.array_equal(concatenated.coefficients[1], coefficient_1)
    assert np.array_equal(concatenated.coefficients[2], coefficient_2)


def test_concatenate_horizontal_non_square():
    mpoly1 = MatrixPolynomial([[1]], [np.ones((2, 3))])
    mpoly2 = MatrixPolynomial([[2]], [np.eye(2)])
    polynomials = [mpoly1, mpoly2]

    concatenated = pa.concatenate(polynomials, axis=1)

    coefficient_0 = np.concatenate([np.ones((2, 3)), np.zeros((2, 2))], axis=1)
    coefficient_1 = np.concatenate([np.zeros((2, 3)), np.eye(2)], axis=1)

    assert concatenated.n_vars == 1
    assert concatenated.degree == 2
    assert np.array_equal(concatenated.exponents, [[1], [2]])
    assert np.array_equal(concatenated.coefficients[0], coefficient_0)
    assert np.array_equal(concatenated.coefficients[1], coefficient_1)


def test_concatenate_vertical_non_square():
    mpoly1 = MatrixPolynomial([[0, 0]], [np.arange(6).reshape(3, 2)])
    mpoly2 = MatrixPolynomial([[2, 1]], [np.ones((4, 2))])
    polynomials = [mpoly1, mpoly2]

    concatenated = pa.concatenate(polynomials)

    coefficient_0 = np.concatenate([np.arange(6).reshape(3, 2), np.zeros((4, 2))])
    coefficient_1 = np.concatenate([np.zeros((3, 2)), np.ones((4, 2))])

    assert concatenated.n_vars == 2
    assert concatenated.degree == 3
    assert np.array_equal(concatenated.exponents, [[0, 0], [2, 1]])
    assert np.array_equal(concatenated.coefficients[0], coefficient_0)
    assert np.array_equal(concatenated.coefficients[1], coefficient_1)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "input_sequence,expected_exception",
    [
        # element with wrong type
        ([MatrixPolynomial([[0]], [np.eye(2)]), np.eye(2)], TypeError),
        # empty sequence
        ([], ValueError),
        # empty sequence
        ((), ValueError),
    ],
)
def test_concatenate_exceptions(axis, input_sequence, expected_exception):
    with pytest.raises(expected_exception):
        pa.concatenate(input_sequence, axis=axis)


@pytest.mark.parametrize(
    "axis,expected_exception",
    [
        # non integer axis
        (0.0, TypeError),
        # non integer axis
        (1.0, TypeError),
        # incorrect axis value
        (-1, ValueError),
        # incorrect axis value
        (2, ValueError),
    ],
)
def test_concatenate_axis_exceptions(axis, expected_exception):
    mpoly = MatrixPolynomial([[0, 0]], [np.eye(2)])

    with pytest.raises(expected_exception):
        pa.concatenate([mpoly], axis=axis)


@pytest.mark.parametrize(
    "axis,shape",
    [
        (0, (2, 3)),
        (0, (1, 1)),
        (1, (3, 2)),
        (1, (1, 1)),
    ],
)
def test_concatenate_shape_exceptions(axis, shape):
    mpoly1 = MatrixPolynomial([[0]], [np.eye(2)])
    mpoly2 = MatrixPolynomial([[0]], [np.ones(shape)])

    with pytest.raises(ValueError):
        pa.concatenate([mpoly1, mpoly2], axis=axis)


def test_block_as_horizontal_concatenate():
    mpoly = MatrixPolynomial([[0]], [np.eye(2)])
    polynomials_grid = [[mpoly, mpoly, mpoly]]

    assembled = pa.block(polynomials_grid)
    concatenated = pa.concatenate(polynomials_grid[0], axis=1)

    assert assembled.n_vars == concatenated.n_vars
    assert assembled.degree == concatenated.degree
    assert np.array_equal(assembled.exponents, concatenated.exponents)
    assert np.array_equal(assembled.coefficients, concatenated.coefficients)


def test_block_as_vertical_concatenate():
    mpoly = MatrixPolynomial([[0]], [np.eye(2)])
    polynomials_grid = [[mpoly], [mpoly], [mpoly]]

    assembled = pa.block(polynomials_grid)
    concatenated = pa.concatenate([mpoly, mpoly, mpoly])

    assert assembled.n_vars == concatenated.n_vars
    assert assembled.degree == concatenated.degree
    assert np.array_equal(assembled.exponents, concatenated.exponents)
    assert np.array_equal(assembled.coefficients, concatenated.coefficients)


def test_block():
    mpoly1 = MatrixPolynomial(
        [[0, 0], [1, 0], [0, 1]], [np.eye(3), np.ones((3, 3)), np.tri(3)]
    )
    mpoly2 = MatrixPolynomial([[2]], [np.arange(6).reshape(3, 2)])
    mpoly3 = MatrixPolynomial([[1, 0], [0, 1]], [3 * np.ones((5, 3)), np.eye(5, 3)])
    mpoly4 = MatrixPolynomial([[0]], [np.arange(10).reshape(5, 2)])

    assembled = pa.block([[mpoly1, mpoly2], [mpoly3, mpoly4]])

    coefficient_0 = np.block(
        [
            [np.eye(3), np.zeros((3, 2))],
            [np.zeros((5, 3)), np.arange(10).reshape(5, 2)],
        ]
    )
    coefficient_1 = np.block(
        [
            [np.ones((3, 3)), np.zeros((3, 2))],
            [3 * np.ones((5, 3)), np.zeros((5, 2))],
        ]
    )
    coefficient_2 = np.block(
        [
            [np.zeros((3, 3)), np.arange(6).reshape(3, 2)],
            [np.zeros((5, 3)), np.zeros((5, 2))],
        ]
    )
    coefficient_3 = np.block(
        [
            [np.tri(3), np.zeros((3, 2))],
            [np.eye(5, 3), np.zeros((5, 2))],
        ]
    )

    assert assembled.n_vars == 2
    assert assembled.degree == 2
    assert np.array_equal(assembled.exponents, [[0, 0], [1, 0], [2, 0], [0, 1]])
    assert np.array_equal(assembled.coefficients[0], coefficient_0)
    assert np.array_equal(assembled.coefficients[1], coefficient_1)
    assert np.array_equal(assembled.coefficients[2], coefficient_2)
    assert np.array_equal(assembled.coefficients[3], coefficient_3)


@pytest.mark.parametrize(
    "input_sequence",
    [
        # inner element is not a sequence
        [[MatrixPolynomial([[0]], [np.eye(1)])], np.eye(1)],
        # inner element is not a sequence
        [[MatrixPolynomial([[0]], [np.eye(1)])], MatrixPolynomial([[0]], [np.eye(1)])],
        # inner element of wrong type
        [[MatrixPolynomial([[0]], [np.eye(1)])], [np.eye(1)]],
    ],
)
def test_block_exceptions(input_sequence):
    with pytest.raises(TypeError):
        pa.block(input_sequence)
