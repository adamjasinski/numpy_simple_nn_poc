import src.basic_algos as alg
import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

# NB - use assert_alclose instead of assert_array_equal for floats
# See https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_array_equal.html

@pytest.mark.parametrize('input,expected', [
    (0.5, 0.6224593312018546),
    (np.array([-1, 0, 1]), np.array([0.26894142, 0.5, 0.73105858])),
])
def test_sigmoid(input, expected):
   actual = alg.sigmoid(input)
   assert_allclose(actual, expected)

@pytest.mark.parametrize('input,expected', [
    ([[2.0, 1.0, 0.1]], [[0.65900114, 0.24243297, 0.09856589]]),
    ([[4.0, 0.5, 0.1]], [[0.95198267, 0.02874739, 0.01926995]]),
])
def test_softmax(input, expected):
   actual = alg.softmax(np.array(input))
   assert_allclose(actual, np.array(expected), rtol=1e-6)