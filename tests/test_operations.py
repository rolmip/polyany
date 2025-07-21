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
