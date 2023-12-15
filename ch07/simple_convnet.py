import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class SimpleConvNet:

    def __init__(self, input_dim: tuple=(1, 28, 28),
                 conv_param: dict={'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 hidden_size: int=100, output_size: int=10, weight_init_std: float=0.01):
        """Initialize a simple convolutional neural network.

        Args:
            input_dim (tuple, optional): Dimensions of the input data (channels, height, width). Defaults to (1, 28, 28).
            conv_param (_type_, optional): Parameters for the convolutional layer. Defaults to {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1}.
                                           - filter_num (int): Number of filters.
                                           - filter_size (int): Size of each filter.
                                           - pad (int): Padding size.
                                           - stride (int): Stride value.
            hidden_size (int, optional): Size of the hidden layer. Defaults to 100.
            output_size (int, optional): Size of the output layer. Defaults to 10.
            weight_init_std (float, optional): Standard deviation for weight initialization. Defaults to 0.01.
        """
        # Extract convolutional layer parameters
        filter_num, filter_size, filter_pad, filter_stride = (
            conv_param['filter_num'], conv_param['filter_size'],
            conv_param['pad'], conv_param['stride']
        )

        # Calculate convolution and pooling output sizes
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) // filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        # Initialize parameters
        self.params = {
            'W1': weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size),
            'b1': np.zeros(filter_num),
            'W2': weight_init_std * np.random.randn(pool_output_size, hidden_size),
            'b2': np.zeros(hidden_size),
            'W3': weight_init_std * np.random.randn(hidden_size, output_size),
            'b3': np.zeros(output_size),
        }

        # Initialize layers
        self.layers = OrderedDict({
            'Conv1': Convolution(self.params['W1'], self.params['b1'], filter_stride, filter_pad),
            'Relu1': Relu(),
            'Pool1': Pooling(pool_h=2, pool_w=2, stride=2),
            'Affine1': Affine(self.params['W2'], self.params['b2']),
            'Relu2': Relu(),
            'Affine2': Affine(self.params['W3'], self.params['b3']),
        })

        self.last_layer = SoftmaxWithLoss()
        
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass to make predictions.

        Args:
            x (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: Predicted probabilities.
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
        return self.last_layer.forward(self.predict(x), t)
    
    def accuracy(self, x: np.ndarray, t: np.ndarray, batch_size: int=100) -> float:
        """Calculate the accuracy for a given input and target.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels.
            batch_size (int, optional): Batch size for accuracy calculation. Defaults to 100.

        Returns:
            float: Accuracy value.
        """
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0
        num_batches = x.shape[0] // batch_size

        for i in range(num_batches):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            y = self.predict(x[start_idx:end_idx])
            acc += np.sum(np.argmax(y, axis=1) == t[start_idx:end_idx])

        return acc / x.shape[0]
    
    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """Calculate numerical gradients for all learnable parameters.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels.

        Returns:
            dict: Gradients for each learnable parameter.
        """
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """Perform the backward pass to calculate gradients for all learnable parameters.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels.

        Returns:
            dict: Gradients for each learnable parameter.
        """
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
    
    def save_params(self, file_name: str="params.pkl"):
        """Save the learnable parameters to a file.

        Args:
            file_name (str, optional): File name for saving parameters. Defaults to "params.pkl".
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name: str="params.pkl"):
        """Load the learnable parameters from a file.

        Args:
            file_name (str, optional): File name for loading parameters. Defaults to "params.pkl".
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]