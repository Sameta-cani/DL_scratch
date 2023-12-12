# coding: utf-8
import sys, os
sys.path.append(os.pardir) # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient

class MultiLayerNetExtend:
    """
    Multi-layer neural network with extended features such as batch normalization, dropout, and weight decay.

    Args:
        input_size (int): Size of the input layer.
        hidden_size_list (list): List of integers representing the sizes of hidden layers.
        output_size (int): Size of the output layer.
        activation (str, optional): Activation function for hidden layers. Defaults to 'relu'.
        weight_init_std (str or float, optional): Standard deviation for weight initialization.
            If 'relu' or 'he' is provided, it uses a scale factor for ReLU. If 'sigmoid' or 'xavier' is provided,
            it uses a scale factor for Sigmoid. Defaults to 'relu'.
        weight_decay_lambda (float, optional): Strength of L2 regularization (weight decay). Defaults to 0.
        use_dropout (bool, optional): Flag to use dropout regularization. Defaults to False.
        dropout_ration (float, optional): Dropout ratio if use_dropout is True. Defaults to 0.5.
        use_batchnorm (bool, optional): Flag to use batch normalization. Defaults to False.

    Attributes:
        input_size (int): Size of the input layer.
        output_size (int): Size of the output layer.
        hidden_size_list (list): List of integers representing the sizes of hidden layers.
        hidden_layer_num (int): Number of hidden layers.
        use_dropout (bool): Flag to use dropout regularization.
        weight_decay_lambda (float): Strength of L2 regularization (weight decay).
        use_batchnorm (bool): Flag to use batch normalization.
        params (dict): Dictionary to store network parameters (weights and biases).
        layers (OrderedDict): Ordered dictionary to store network layers in sequence.
        last_layer (SoftmaxWithLoss): Softmax with Loss layer for the output layer.
    """
    def __init__(self, input_size:int, hidden_size_list: list, output_size: int,
                 activation: str = 'relu', weight_init_std: str or float = 'relu', weight_decay_lambda: float = 0, 
                 use_dropout: bool = False, dropout_ratio: float = 0.5, use_batchnorm: bool = False):
        """
        Initialize the MultiLayerNetExtend.

        Args are used to set up the architecture of the neural network and its training parameters.
        Weights and biases are initialized based on the provided weight_init_std.

        Parameters:
            input_size (int): Size of the input layer.
            hidden_size_list (list): List of integers representing the sizes of hidden layers.
            output_size (int): Size of the output layer.
            activation (str, optional): Activation function for hidden layers. Defaults to 'relu'.
            weight_init_std (str or float, optional): Standard deviation for weight initialization.
                If 'relu' or 'he' is provided, it uses a scale factor for ReLU.
                If 'sigmoid' or 'xavier' is provided, it uses a scale factor for Sigmoid.
                Defaults to 'relu'.
            weight_decay_lambda (float, optional): Strength of L2 regularization (weight decay). Defaults to 0.
            use_dropout (bool, optional): Flag to use dropout regularization. Defaults to False.
            dropout_ration (float, optional): Dropout ratio if use_dropout is True. Defaults to 0.5.
            use_batchnorm (bool, optional): Flag to use batch normalization. Defaults to False.
        """
        # Initialization of network parameters
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.activation = activation
        self.weight_init_std = weight_init_std,
        self.weight_decay_lambda = weight_decay_lambda
        self.use_dropout = use_dropout
        self.dropout_ratio = dropout_ratio
        self.use_batchnorm = use_batchnorm
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}

        # Weight initialization
        self.__init_weight(weight_init_std)

        # Activation layer based on the specified activation function
        activation_layer = {'sigmoid': Sigmoid, 'relu': Relu, 'selu': Selu, 'elu': ELU}
        self.layers = OrderedDict()
        
        # Hidden layers
        for idx in range(1, self.hidden_layer_num + 1):
            self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])
            if self.use_batchnorm:
                self.params[f'gamma{idx}'] = np.ones(hidden_size_list[idx-1])
                self.params[f'beta{idx}'] = np.zeros(hidden_size_list[idx-1])
                self.layers[f'BatchNorm{idx}'] = BatchNormalization(self.params[f'gamma{idx}'], self.params[f'beta{idx}'])
            self.layers[f'Activation_function{idx}'] = activation_layer[activation]()

            # Dropout layer
            if self.use_dropout:
                self.layers[f'Dropout{idx}'] = AlphaDropout(dropout_ratio)

        # Output layer
        idx = self.hidden_layer_num + 1
        self.layers[f'Affine{idx}'] = Affine(self.params[f'W{idx}'], self.params[f'b{idx}'])

        # Softmax with Loss layer for the output
        self.last_layer = SoftmaxWithLoss()

    def __init_weight(self, weight_init_std: str or float):
        """
        Initialize weights and biases for each layer based on the specified weight initialization method.

        Args:
            weight_init_std (str or float): Standard deviation for weight initialization.
                If 'relu' or 'he' is provided, it uses a scale factor for ReLU.
                If 'sigmoid' or 'xavier' is provided, it uses a scale factor for Sigmoid.
                If a float is provided, it uses that value as the standard deviation.
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]

        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he', 'elu'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier', 'selu'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])

            self.params[f'W{idx}'] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params[f'b{idx}'] = np.zeros(all_size_list[idx])

    def predict(self, x: np.array, train_flg: bool = False) -> np.array:
        """
        Forward pass to make predictions.

        Args:
            x (np.array): Input data.
            train_flg (bool, optional): Flag indicating whether the model is in training mode. Defaults to False.

        Returns:
            np.array: Predicted output.
        """
        for layer in self.layers.values():
            if "Dropout" in layer.__class__.__name__ or "BatchNorm" in layer.__class__.__name__:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x

    def loss(self, x: np.array, t: np.array, train_flg: bool = False) -> float:
        """
        Compute the loss.

        Args:
            x (np.array): Input data.
            t (np.array): Ground truth labels.
            train_flg (bool, optional): Flag indicating whether the model is in training mode. Defaults to False.

        Returns:
            float: Loss value.
        """
        y = self.predict(x, train_flg)

        # Weight decay term
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params[f'W{idx}']
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        # Cross-entropy loss + Weight decay
        return self.last_layer.forward(y, t) + weight_decay

    def accuracy(self, X: np.array, T: np.array) -> float:
        """
        Compute the accuracy of the model on the given data.

        Args:
            X (np.array): Input data.
            T (np.array): Ground truth labels.

        Returns:
            float: Accuracy.
        """
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1:
            T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy

    def numerical_gradient(self, X: np.array, T: np.array) -> float:
        """
        Compute the numerical gradient of the loss with respect to the parameters.

        Args:
            X (np.array): Input data.
            T (np.array): Ground truth labels.

        Returns:
            dict: Dictionary containing numerical gradients for each parameter.
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = numerical_gradient(loss_W, self.params[f'W{idx}'])
            grads[f'b{idx}'] = numerical_gradient(loss_W, self.params[f'b{idx}'])

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads[f'gamma{idx}'] = numerical_gradient(loss_W, self.params[f'gamma{idx}'])
                grads[f'beta{idx}'] = numerical_gradient(loss_W, self.params[f'beta{idx}'])

        return grads

    def gradient(self, x: np.array, t: np.array) -> dict:
        """
        Compute the gradient of the loss with respect to the parameters using backpropagation.

        Args:
            x (np.array): Input data.
            t (np.array): Ground truth labels.

        Returns:
            dict: Dictionary containing gradients for each parameter.
        """
        # Forward pass and loss computation
        self.loss(x, t, train_flg=True)

        # Backward pass
        dout = 1
        dout = self.last_layer.backward(dout)

        # Reverse the order of layers for backward pass
        layers = list(self.layers.values())
        layers.reverse()

        # Backward pass through all layers
        for layer in layers:
            dout = layer.backward(dout)

        # Compute gradients for each parameter
        grads = {}
        for idx in range(1, self.hidden_layer_num + 2):
            grads[f'W{idx}'] = self.layers[f'Affine{idx}'].dW + self.weight_decay_lambda * self.params[f'W{idx}']
            grads[f'b{idx}'] = self.layers[f'Affine{idx}'].db

            if self.use_batchnorm and idx != self.hidden_layer_num + 1:
                grads[f'gamma{idx}'] = self.layers[f'BatchNorm{idx}'].dgamma
                grads[f'beta{idx}'] = self.layers[f'BatchNorm{idx}'].dbeta

        return grads