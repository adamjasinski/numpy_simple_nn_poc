from src.neural import Neural
import numpy as np
import pytest

def test_neural_prediction_without_train_should_fail():
    nn = Neural(input_size=10, hidden1_size=2, hidden2_size=2, output_size=2)
    with pytest.raises(Exception, match="^The model has not been trained yet"):
        nn.predict(np.zeros((10,10)))