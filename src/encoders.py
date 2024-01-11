import numpy as np

class OneHotEncoder:
    @staticmethod
    def encode(y, num_labels):
        """
        Convert class labels to one-hot encoded vectors.

        This function takes an array of class labels and converts them into one-hot encoded
        vectors. Each one-hot encoded vector represents the presence of a class label using a
        1.0 in the corresponding position and 0.0 elsewhere.

        Parameters:
        - y (array-like): An array of class labels to be one-hot encoded.
        - num_labels (int, optional): The total number of unique class labels. Defaults to 10.

        Returns:
        - one_hot (numpy.ndarray): A 2D numpy array where each column is a one-hot encoded
        vector representing a class label.

        Example usage:
        >>> y = [0, 2, 1, 3, 0]
        >>> one_hot_enc(y, num_labels=4)
        array([[1., 0., 0., 0.],
            [0., 0., 1., 0.],
            [0., 1., 0., 0.],
            [0., 0., 0., 1.],
            [1., 0., 0., 0.]])
        """
        #one_hot = np.zeros((len(y), num_labels))
        #for i, num in enumerate(y):
        #    one_hot[i, num] = 1
        one_hot = np.eye(num_labels)[y]
        return one_hot
    
    @staticmethod
    def decode(y_one_hot):
        if not isinstance(y_one_hot, np.ndarray) or y_one_hot.ndim != 2:
            raise ValueError("Expected a 2 dimensional array")
        
        row_num = y_one_hot.shape[0]
        result = np.zeros(row_num)
        for i in range(0, row_num):
            row = y_one_hot[i,:]
            result[i] = np.argmax(row)

        return result