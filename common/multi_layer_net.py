import sys, os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNet:
    def __init__(self, input_size: int, hidden_size_list: list, output_size: int,
                 activation: str='relu', weight_init_std: str='relu', weight_decay_lambda: float=0):
        """Initialize a Multi-Layer Neural Network.

        Args:
            input_size (int): Dimensionality of the input.
            hidden_size_list (list): List of hidden layer sizes.
            output_size (int): Dimensionality of the output.
            activation (str, optional): Activation function for hidden layers ('relu' or 'sigmoid'). Defaults to 'relu'.
            weight_init_std (str, optional): Weight initialization method ('relu'/'he' or 'simgoid'/'xavier'). Defaults to 'relu'.
            weight_decay_lambda (float, optional): Strength of L2 regularization (weight decay). Defaults to 0.
        """
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.weight_decay_lambda = weight_decay_lambda
        self.params = {}

        self.__init_weight(weight_init_std)

        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu}
        self.layers = OrderedDict()

        for idx, hidden_size in enumerate(hidden_size_list, start=1):
            self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])
            self.layers[f'Activation_function{idx}'] = activation_layer[activation]()

        last_idx = self.hidden_layer_num + 1
        self.layers[f'Affine{last_idx}'] = Affine(self.params[f'W{last_idx}'], self.params[f'b{last_idx}'])
        
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std: str):
        """Initialize weight and biases for all layers.

        Args:
            weight_init_std (str): Weight initialization method ('relu'/'he' or 'sigmoid'/'xavier').
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])
            self.params[f'W{idx}'] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params[f'b{idx}'] = np.zeros(all_size_list[idx])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass to make predictions.

        Args:
            x (np.ndarray): Input data.

        Returns:
            np.ndarray: Predicted output.
        """
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    
    def loss(self, x: np.ndarray, t: np.ndarray) -> float:
        y = self.predict(x)
        regularization_term = 0.5 * self.weight_decay_lambda * sum(np.sum(self.params[f'W{idx}'] ** 2) for idx in range(1, self.hidden_layer_num + 2))

        return self.last_layer.forward(y, t) + regularization_term
    
    def accuracy(self, x: np.ndarray, t: np.ndarray) -> float:
        """Compute the accuracy for a given input and target.

        Args:
            x (np.ndarray): Input data.
            t (np.ndarray): Target labels.

        Returns:
            float: Accuracy value.
        """
        y = np.argmax(self.predict(x), axis=1)
        t = np.argmax(t, axis=1) if t.ndim != 1 else t

        accuracy = np.sum(y == t) / len(t)
        return accuracy
    
    def numerical_gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """Compute numerical gradients of the loss with respect to parameters.

        Args:
            x (np.ndarray): Input data.
            t (np.ndarray): Target labels.

        Returns:
            dict: Gradients for each parameter.
        """
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = numerical_gradient(loss_W, self.params[f'W{idx}'])
            grads[f'b{idx}'] = numerical_gradient(loss_W, self.params[f'b{idx}'])

        return grads
    
    def gradient(self, x: np.ndarray, t: np.ndarray) -> dict:
        """Compute gradients of the loss with respect to parameters using backpropagation.

        Args:
            x (np.ndarray): Input data.
            t (np.ndarray): Target labels.

        Returns:
            dict: Gradients for each parameter.
        """
        self.loss(x, t)

        dout = 1
        dout = self.last_layer.backward(dout)

        for layer in reversed(self.layers.values()):
            dout = layer.backward(dout)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            affine_layer = self.layers[f'Affine{idx}']
            grads[f'W{idx}'] = affine_layer.dW + self.weight_decay_lambda * affine_layer.W
            grads[f'b{idx}'] = affine_layer.db

        return grads