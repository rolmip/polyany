import operator

import numpy as np
import pytest

from polyany import Polynomial


@pytest.mark.parametrize(
    "input_data,expected_output",
    [
        ([0, 0, 0], -2),
        ([4.5859, -0.8247, 0.8993], -29.077021995623053),
        ([-3.7941, 2.0937, -1.6209], -254.67936923907425),
        ([-1.164, 2.1334, -3.4052], -86.14505044340349),
        ([-0.0875, -2.2518, 1.3716], -17.295310852174996),
        ([1.9793, -4.477, -1.6816], 299.4298549970985),
    ],
)
def test_polynomial_eval(input_data, expected_output):
    poly = Polynomial([[0, 0, 0], [0, 1, 0], [1, 2, 0], [2, 0, 2]], [-2, 5, 9, -3])

    assert np.isclose(poly(input_data), expected_output)


@pytest.mark.parametrize(
    "input_data,expected_exception",
    [
        # string input
        ("polyany", TypeError),
        # none input
        (None, TypeError),
        # scalar input
        (0, ValueError),
        # wrong number of components
        ([0, 0], ValueError),
    ],
)
def test_polynomial_eval_exceptions(input_data, expected_exception):
    with pytest.raises(expected_exception):
        poly = Polynomial([[0, 0, 0], [0, 1, 0], [1, 2, 0], [2, 0, 2]], [-2, 5, 9, -3])
        poly(input_data)


def test_polynomial_equality_true():
    poly1 = Polynomial.univariate([1, 2, 3])
    poly2 = Polynomial([[0], [1], [2]], [1, 2, 3])

    assert poly1 == poly2


def test_polynomial_equality_false():
    poly1 = Polynomial.univariate([1, 2, 3])
    poly2 = Polynomial.univariate([10, 20, 30])

    assert poly1 != poly2


def test_polynomial_equality_different_order():
    exponents1 = [[0, 0, 0], [1, 1, 2], [3, 1, 0], [0, 0, 2]]
    coefficients1 = [1, 2, 3, 4]
    poly1 = Polynomial(exponents1, coefficients1)

    exponents2 = [[3, 1, 0], [0, 0, 0], [0, 0, 2], [1, 1, 2]]
    coefficients2 = [3, 1, 4, 2]
    poly2 = Polynomial(exponents2, coefficients2)

    assert poly1 == poly2


def test_polynomial_equality_different_degree():
    poly1 = Polynomial([[0, 0], [1, 0], [0, 1]], [0, 1, 2])
    poly2 = Polynomial([[0, 0], [1, 0], [0, 2]], [0, 1, 2])

    assert poly1 != poly2


def test_polynomial_equality_different_n_vars():
    poly1 = Polynomial.univariate([1, 2, 3])
    poly2 = Polynomial([[0, 0], [0, 1], [0, 2]], [1, 2, 3])

    assert poly1 != poly2


@pytest.mark.parametrize(
    "input_data",
    [
        1,
        "polyany",
        None,
        [1, 2, 3],
        (1, 2, 3),
    ],
)
def test_polynomial_equality_different_types(input_data):
    poly = Polynomial.univariate([1, 2, 3])

    assert poly != input_data


def test_polynomial_equality_ndarrays():
    poly = Polynomial.univariate([1, 2, 3])
    array = np.array([1, 2, 3])

    assert not np.all(poly == array)


@pytest.mark.parametrize(
    "operation", [operator.lt, operator.le, operator.gt, operator.ge]
)
@pytest.mark.parametrize(
    "other", [1, "polyany", None, [1, 2, 3], (1, 2, 3), np.array([1, 2, 3])]
)
def test_polynomial_ordering_exceptions(operation, other):
    poly = Polynomial.univariate([1, 2, 3])
    with pytest.raises(TypeError):
        operation(poly, other)


@pytest.mark.parametrize("input_data", (np.array(1), "polyany", None, [1]))
def test_polynomial_shift_non_int_input(input_data):
    poly = Polynomial.univariate([1, 2, 3])
    with pytest.raises(TypeError):
        poly.shift(input_data)


@pytest.mark.parametrize("operation", [operator.lshift, operator.rshift])
def test_polynomial_shift_wrapper_negative_input(operation):
    poly = Polynomial.univariate([1, 2, 3])
    with pytest.raises(ValueError):
        operation(poly, -1)


@pytest.mark.parametrize(
    "input_k",
    (
        # has nonzero coefficients
        -1,
        # at least one variable must remain
        -2,
    ),
)
def test_polynomial_shift_invalid_left_shift(input_k):
    poly = Polynomial([[0, 0], [0, 1], [1, 0]], [10, 20, 30])
    with pytest.raises(ValueError):
        poly.shift(input_k)


def test_polynomial_right_shift_inverse():
    poly = Polynomial.univariate([1, 2, 3])

    assert poly.shift(1).shift(-1) == poly


def test_polynomial_left_shift_inverse():
    poly = Polynomial([[0, 0], [0, 3], [0, 5]], [10, 20, 30])

    assert poly.shift(-1).shift(1) == poly


def test_polynomial_neg():
    poly = Polynomial.univariate([1, -2, 3])
    neg_poly = Polynomial.univariate([-1, 2, -3])

    assert -poly == neg_poly


@pytest.mark.parametrize(
    "scalar,expected_coefficient",
    [
        (-1, [0, 2, 3]),
        (0, [1, 2, 3]),
        (1, [2, 2, 3]),
    ],
)
def test_polynomial_add_scalar_with_constant_term(scalar, expected_coefficient):
    poly = Polynomial.univariate([1, 2, 3])
    expected = Polynomial.univariate(expected_coefficient)

    assert (poly + scalar) == expected


@pytest.mark.parametrize(
    "scalar,expected_coefficient",
    [
        (-1, [-1, 1, 2]),
        (0, [0, 1, 2]),
        (1, [1, 1, 2]),
    ],
)
def test_polynomial_add_scalar_without_constant_term(scalar, expected_coefficient):
    poly = Polynomial([[1, 0], [0, 1]], [1, 2])
    expected = Polynomial([[0, 0], [1, 0], [0, 1]], expected_coefficient)

    assert (poly + scalar) == expected


@pytest.mark.parametrize(
    "scalar,expected_coefficient",
    [
        (-1, [0, 2, 3]),
        (0, [1, 2, 3]),
        (1, [2, 2, 3]),
    ],
)
def test_polynomial_reflected_add_scalar(scalar, expected_coefficient):
    poly = Polynomial.univariate([1, 2, 3])
    expected = Polynomial.univariate(expected_coefficient)

    assert (scalar + poly) == expected
