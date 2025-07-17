import numpy as np
import pytest

from polyany.polynomial import Polynomial


@pytest.mark.parametrize(
    "input_data,expected_string",
    [
        (
            (
                np.array([[1, 2, 0], [0, 1, 0], [0, 0, 0], [3, 0, 2]]),
                np.array([3, 4, 10, 2]),
            ),
            "10 + 4*x_2 + 3*x_1*x_2^2 + 2*x_1^3*x_3^2",
        ),
        (
            (np.array(range(4)).reshape(-1, 1), np.array([13, 11.5, 2, 1.333])),
            "13 + 11.5*x_1 + 2*x_1^2 + 1.333*x_1^3",
        ),
        (
            (np.array([[0, 0], [1, 0], [0, 1], [1, 1]]), np.array([0, -1, 1, -1])),
            "-x_1 + x_2 - x_1*x_2",
        ),
        (
            (
                np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                np.array([0, 0, 0, 0]),
            ),
            "0",
        ),
    ],
)
def test_polynomial_string_representation(input_data, expected_string):
    poly = Polynomial(*input_data)

    assert str(poly) == expected_string


@pytest.mark.parametrize(
    "input_data,expected_exception",
    [
        # string input as exponent
        (("polyany", [[1, 2, 3]]), TypeError),
        # string input as coefficient
        (([[1, 2], [0, 1], [0, 0]], "polyany"), TypeError),
        # none input
        (([[1, 2], [0, 1], [0, 0]], None), TypeError),
        # float exponents (non safe-convertible to int)
        (([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]], [[-1, 2, 33]]), TypeError),
        # exponents without 2 dimensions
        (([1, 2, 3], [10, 11, 12]), ValueError),
        # coefficients without 1 dimension
        (([[0, 0], [1, 0], [0, 1]], [[1, 2], [3, 4]]), ValueError),
        # coefficients without 1 dimension
        (([[0, 0], [1, 0], [0, 1]], [[1, 2, 3]]), ValueError),
        # scalar coefficient
        (([[0, 0]], 0), ValueError),
        # non unique exponents
        (([[0, 0], [0, 0]], [1, 2]), ValueError),
        # different number of exponents and coefficients
        (([[1, 2], [0, 0], [1, 1], [2, 1]], [10, 11, 12]), ValueError),
        # nonlinear polynomial (negative exponents)
        (([[0, 0], [-1, 0], [0, -1]], [-1, 1, -1]), ValueError),
    ],
)
def test_polynomial_creation_exceptions(input_data, expected_exception):
    with pytest.raises(expected_exception):
        Polynomial(*input_data)


@pytest.mark.parametrize(
    "input_data,expected_string",
    [
        ([1], "1"),
        ([-1], "-1"),
        ([1, 2, 3], "1 + 2*x_1 + 3*x_1^2"),
    ],
)
def test_polynomial_univariate(input_data, expected_string):
    poly = Polynomial.univariate(input_data)

    assert str(poly) == expected_string


@pytest.mark.parametrize(
    "input_data,expected_exception",
    [
        # scalar input
        (1, ValueError),
        # coefficients without 1 dimension
        ([[1]], ValueError),
    ],
)
def test_polynomial_univariate_exceptions(input_data, expected_exception):
    with pytest.raises(expected_exception):
        Polynomial.univariate(input_data)


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
