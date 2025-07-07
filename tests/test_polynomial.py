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
