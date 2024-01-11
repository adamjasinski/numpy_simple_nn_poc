import numpy as np

def sigmoid(x):
    """Compute the sigmoid activation function.

    The sigmoid activation function is commonly used in machine learning and neural networks
    to map real-valued numbers to values between 0 and 1. 

    Parameters:
    - x (float, array-like): The input value(s) to apply the sigmoid function to.

    Returns:
    - value (np.ndarray): It returns the sigmoid of the input 'x'.

    Example usage:
    >>> sigmoid(0.5)
    0.6224593312018546

    >>> sigmoid(np.array([-1, 0, 1]), derivative=True)
    [0.19661193 0.25       0.19661193]

    """
    value = 1 / (1 + np.exp(-x))
    return value

def softmax(x):
    """
    Compute the softmax activation function.

    The softmax activation function is commonly used in machine learning and neural networks
    to convert a vector of real numbers into a probability distribution over multiple classes.
    It exponentiates each element of the input vector and normalizes it to obtain the probabilities.

    Parameters:
    - x (numpy.ndarray): The input vector to apply the softmax function to.

    Returns:
    - value (np.ndarray): It returns the softmax of the input 'x', which is a probability distribution.

    Example usage:
    >>> softmax([2.0, 1.0, 0.1])
    [0.65900114 0.24243297 0.09856589]

    >>> softmax([4.0, 0.5, 0.1])
    [0.22471864 0.18365923 0.08885066]
    """
    # Numerically stable with large exponentials
    # xmax = np.max(x)
    # nom = np.exp(x-xmax)
    # denom = np.sum(np.exp(x-xmax))
    # value = nom / denom
    # return value

    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)