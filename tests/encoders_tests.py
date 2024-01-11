from src.encoders import OneHotEncoder
import numpy as np
from numpy.testing import assert_array_equal

def test_encode():
    input = [0, 2, 1, 3, 0]
    expected =  np.array([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]])
    
    actual = OneHotEncoder.encode(input, num_labels=4)
    assert_array_equal(actual, expected)

def test_decode():
    input =  np.array([
        [1., 0., 0., 0.],
        [0., 0., 1., 0.],
        [0., 1., 0., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]])
    expected = [0, 2, 1, 3, 0]
    assert input.ndim == 2
    assert input.shape[0] == 5
    actual = OneHotEncoder.decode(input)
    assert_array_equal(actual, expected)