from typing import Callable
import numpy as np
from . import basic_algos as alg

class Neural:
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        """Initialize a neural network"""
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

    def train(self, X_train, y_train, learning_rate = 0.1, epochs = 10, batch_size = 64, progress_func=None):
        """
        Train the neural network by running the backpropagation algorith
        
        Parameters:
        - X_train (numpy.ndarray): Input data (observations)
        - y_train (numpy.ndarray): Target data
        - learning_rate (float): Learning rate
        - epochs (int): Number of iterations of the algorithm
        - batch_size (int): Batch size in each step of the algorithm
        - progress_func (Callable): callback function being called after iteration of the algorithm
        """
        # Training loop
        for epoch in range(epochs):
            for i in range(0, len(X_train), batch_size):
                x_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                # Forward pass
                z1 = np.dot(x_batch, self.W1) + self.b1
                a1 = alg.sigmoid(z1)
                z2 = np.dot(a1, self.W2) + self.b2
                a2 = alg.sigmoid(z2)
                z3 = np.dot(a2, self.W3) +self. b3
                a3 = alg.softmax(z3)

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
            a1 = alg.sigmoid(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            a2 = alg.sigmoid(z2)
            z3 = np.dot(a2, self.W3) + self.b3
            a3 = alg.softmax(z3)

            # predictions = np.argmax(a3, axis=1)
            # train_accuracy = np.mean(predictions == y_train)

            # if progress_func is not None and isinstance(progress_func, Callable):
            #     progress_func(epoch, train_accuracy)

    def score(self, X, y):
        # Calculate accuracy on the training set
        z1 = np.dot(X, self.W1) + self.b1
        a1 = alg.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = alg.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = alg.softmax(z3)

        predictions = np.argmax(a3, axis=1)
        accuracy = np.mean(predictions == y)
        return accuracy

    def predict(self, X):
        # Calculate accuracy on the test set
        z1 = np.dot(X, self.W1) + self.b1
        a1 = alg.sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        a2 = alg.sigmoid(z2)
        z3 = np.dot(a2, self.W3) + self.b3
        a3 = alg.softmax(z3)
        predictions = np.argmax(a3, axis=1)
        return predictions
    
