import numpy as np

class Neural:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        self._data = None
        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        self.W1, self.W2, self.W3, self.b1, self.b2, self.b3 = Neural.initialize_wb(input_size, hidden1_size, hidden2_size, output_size)


    @staticmethod
    def initialize_wb(input_size, hidden1_size, hidden2_size, output_size):
        """
        Initialize weights and biases for a neural network.

        Parameters:
        - input_size (int): Number of input features.
        - hidden1_size (int): Number of neurons in the first hidden layer.
        - hidden2_size (int): Number of neurons in the second hidden layer.
        - output_size (int): Number of output neurons (number of classes).

        Returns:
        - W1, W2, W3 (np.ndarray): 2D arrays for the weights with random values from a standard normal
            distribution
        - b1, b2, b3: (np.ndarray): 2D arrays for the bias as zeros
        
        Example usage:
        W1, W2, W3, b1, b2, b3 = initialize_wb(input_size, hidden1_size, hidden2_size, output_size)
        """

        # Initialize weights with random values from a standard normal distribution
        W1 = np.random.normal(0, 1, size = (input_size, hidden1_size)) # size input x hidden1
        W2 = np.random.normal(0, 1, size = (hidden1_size, hidden2_size)) # size hidden1 x hidden2
        W3 = np.random.normal(0, 1, size = (hidden2_size, output_size)) # size hidden2 x output
        
        # Initialize biases as zeros
        b1 = np.zeros((1, hidden1_size)) # size 1 x hidden1
        b2 = np.zeros((1, hidden2_size)) # size 1 x hidden2
        b3 = np.zeros((1, output_size)) # size 1 x output
        return W1, W2, W3, b1, b2, b3

    def train(self, X_train, y_train, learning_rate = 0.1, epochs = 10, batch_size = 64):
        """Train the neural network by running the backpropagation algorith"""
        # Training loop
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i + batch_size]
                y_batch = y_train_onehot[i:i + batch_size]

                # Forward pass
                z1 = np.dot(x_batch, self.W1) + self.b1
                a1 = Neural.sigmoid(z1)
                z2 = np.dot(a1, self.W2) + self.b2
                a2 = Neural.sigmoid(z2)
                z3 = np.dot(a2, self.W3) +self. b3
                a3 = Neural.softmax(z3)

                # Backpropagation
                delta = a3 - y_batch
                dW3 = np.dot(a2.T, delta)
                db3 = np.sum(delta, axis=0, keepdims=True)
                delta = np.dot(delta, self.W3.T) * (a2 * (1 - a2))
                dW2 = np.dot(a1.T, delta)
                db2 = np.sum(delta, axis=0, keepdims=True)
                delta = np.dot(delta, self.W2.T) * (a1 * (1 - a1))
                dW1 = np.dot(x_batch.T, delta)
                db1 = np.sum(delta, axis=0, keepdims=True)

                # Update weights and biases
                self.W1 -= learning_rate * dW1
                self.b1 -= learning_rate * db1
                self.W2 -= learning_rate * dW2
                self.b2 -= learning_rate * db2
                self.W3 -= learning_rate * dW3
                self.b3 -= learning_rate * db3

            # Calculate accuracy on the training set
            z1 = np.dot(X_train, self.W1) + self.b1
            a1 = Neural.sigmoid(z1)
            z2 = np.dot(self.a1, self.W2) + self.b2
            a2 = Neural.sigmoid(z2)
            z3 = np.dot(self.a2, self.W3) + self.b3
            a3 = Neural.softmax(z3)

            predictions = np.argmax(a3, axis=1)
            train_accuracy = np.mean(predictions == y_train)

        return train_accuracy

    def score(self, X, y):
        # Calculate accuracy on the training set
        z1 = np.dot(X, self.W1) + self.b1
        a1 = Neural.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = Neural.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = Neural.softmax(z3)

        predictions = np.argmax(a3, axis=1)
        train_accuracy = np.mean(predictions == y)
        return train_accuracy

    def predict(self, X):
        # Calculate accuracy on the test set
        z1 = np.dot(X, self.W1) + self.b1
        a1 = Neural.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = Neural.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = Neural.softmax(z3)
        predictions = np.argmax(a3, axis=1)
        return predictions
    
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