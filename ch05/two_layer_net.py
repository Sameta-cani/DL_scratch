import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.layers import *
from common.gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNet:
    """A two-layer neural network with optional optimizations.

    Parameters:
        input_size (int): The number of input features.
        hidden_size (int): The number of neurons in the hidden layer.
        output_size (int): The number of output classes.
        weight_init_std (float, optional): The standard deviation for weight initialization. Defaults to 0.01.

    Attributes:
        params (dict): A dictionary to store the network parameters.
        layers (OrderedDict): An ordered dictionary to store the layer of the network.
        lastLayer: The last layer of the network responsible for the loss calculation.
    """
    def __init__(self, input_size: int, hidden_size: int, output_size: int, weight_init_std: float=0.01):
        """Initialize the neural network with specified sizes and optional weight initialization.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output classes.
            weight_init_std (float, optional): The standard deviation for weight initialization. Defaults to 0.01.
        """
        # Initialize parameters
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # Initialize layers
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        
        # Initialize the last layer
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform a forward pass through the network to make predictions.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted output.
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        """Calculate the loss for a given input and target.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels.

        Returns:
            float: Loss value.
        """
        y = self.predict(x)
        return self.lastLayer.forward(y, t)
    
    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        """Calculate the accuracy for a given input and target.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels

        Returns:
            float: Accuracy.
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: 
            t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """Calculate the numerical gradients of the parameters.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels.

        Returns:
            dict: Gradients for each parameter.
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for key in self.params.keys():
            grads[key] = numerical_gradient(loss_W, self.params[key])

        return grads
    
    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """Calculate the gradients of the parameters using backpropagation.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels

        Returns:
            dict: Gradients for each parameter.
        """
        # Forward pass
        self.loss(x, t)

        # Backward pass
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # Gradients
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads