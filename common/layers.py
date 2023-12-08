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
    
class SoftmaxWithLoss:
    """
    The SoftmaxWithLoss class handles softmax and cross-entropy loss, providing method for forward and backward passes.

    Attributes:
        loss (float): Current loss value.
        y (numpy.ndarray): Output of the softmax.
        t (numpy.ndarray): Ground truth labels.

    Methods:
        forward(x, t):
        Computes softmax and loss for the given input x and ground truth labels t, returning the loss.

        backward(dout=1):
            Performs backward propagation to compute gradients with respect to the input and returns them.

    Examples:
        softmax_loss = SoftmaxWithLoss()
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        t = np.array([0, 1]) # Ground truth labels
        loss = softmax_loss.forward(x, t)
        gradient = softmax_loss.backward()
    """
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x: np.ndarray, t: np.ndarray) -> float:
        """Computes softmax and loss for the given input x and ground truth labels t, returning the loss.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Ground truth labels

        Returns:
            float: Computed loss value.
        """
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss
    
    def backward(self, dout: float=1) -> np.ndarray:
        """Performs backward propagation to compute gradients with respect to the input and returns them.

        Args:
            dout (float, optional): Gradient passed from the upper layer during backpropation. Defaults to 1.

        Returns:
            numpy.ndarray: Gradients with respect to the input.
        """
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx