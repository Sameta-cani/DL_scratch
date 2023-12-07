import numpy as np
from common.functions import *

class Relu:
    """Rectified Linear Unit (ReLU) activation function for neural networks.

    This class represents the ReLU activation function used in neural networks.
    It has forward and backward methods to perform forward and backward passes.

    Attributes:
        mask (numpy.ndarray): A boolean mask indicating the locations where the input is less than or equal to 0.
    """
    def __init__(self):
        self.mask = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass and return the result of ReLU activation.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The result of the ReLU activation.
        """
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform backward pass and return the gradient.

        Args:
            dout (numpy.ndarray): The gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: The gradient of the ReLU activation with respect to the input.
        """
        dout[self.mask] = 0
        dx = dout

        return dx
    
class Sigmoid:
    """Sigmoid activation function for neural networks.

    This class represents the Sigmoid activation function used in neural networks.
    It has forward and backward methods to perform forward and backward passes.

    Attributes:
        out (numpy.ndarray): The output of the Sigmoid activation during the forward pass.
    """
    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass and return the result of the Sigmoid activation.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The result of the Sigmoid activation.
        """
        out = sigmoid(x)
        self.out = out
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform backward pass and return the gradient.

        Args:
            dout (numpy.ndarray): The gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: The gradient of the Sigmoid activation with respect to the input.
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
class Affine:
    """Affine layer for neural networks.

    This class represents an Affine layer used in neural networks.
    It has forward and backward methods to perform forward and backward passes.

    Attributes:
        W (numpy.ndarray): The weight matrix.
        b (numpy.ndarray): The bias vector.
        x (numpy.ndarray): The input array.
        original_x_shape (tuple): The original shape of the input array.
        dW (numpy.ndarray): The gradient of the loss with resepect to the weight matrix.
        db (numpy.ndarray): The gradient of the loss with respect to the bias vector.
    """

    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass and return the result of the Affine transformation.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The result of the Affine transformation.
        """
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b
        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform backward pass and return the gradient.

        Args:
            dout (numpy.ndarray): The gradient of the loss with respect to the output.

        Returns:
            numpy.ndarray: The gradient of the Affine transformation with respect to the input.
        """
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)
        return dx