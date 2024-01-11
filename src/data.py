import mnist
import numpy as np
import matplotlib.pyplot as plt

class DataHelper:
  @staticmethod
  def load_and_split_data():
      """
      Load and return the MNIST dataset for training and testing.

      Returns:
      - X_train (numpy.ndarray): Training set, a 3D array of shape (num_samples, 28, 28),
        where num_samples is the number of training samples.
      - y_train (numpy.ndarray): Training labels, a 1D array of shape (num_samples,) containing
        the corresponding class labels for the training images.
      - X_test (numpy.ndarray): Test set, a 3D array of shape (num_samples, 28, 28),
        where num_samples is the number of test samples.
      - y_test (numpy.ndarray): Test labels, a 1D array of shape (num_samples,) containing
        the corresponding class labels for the test images.

      Example usage:
      >>> X_train, y_train, X_test, y_test = data_loader()
      >>> print("Training set shape:", X_train.shape)
      >>> print("Training labels shape:", y_train.shape)
      >>> print("Test set shape:", X_test.shape)
      >>> print("Test labels shape:", y_test.shape)
      """
      X_train = mnist.train_images()
      y_train = mnist.train_labels()
      
      X_test = mnist.test_images()
      y_test = mnist.test_labels()

      return X_train, y_train, X_test, y_test
  
  @staticmethod
  def visualise(X, y):
    fig, axs = plt.subplots(4,4)
    for ax in axs.flatten():
        idx = np.random.randint(0, len(X))
        ax.imshow(X[idx], cmap='gray')
        ax.set_title(y[idx])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout(pad=0)
    return fig
  
  @staticmethod
  def preprocess_images(X):
    X = X / 255.0
    m = X.shape[0]
    # Flatten the images
    return X.reshape(m, -1)