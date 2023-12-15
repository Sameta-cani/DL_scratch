import sys, os
sys.path.append(os.pardir)
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *


class DeepConvNet:
    def __init__(self, input_dim: tuple=(1, 28, 28),
                 conv_param_1: dict = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2: dict = {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3: dict = {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4: dict = {'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5: dict = {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6: dict = {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size: int=50, output_size: int=10):
        """Initialize a deep convolutional neural network.

        Args:
            input_dim (tuple, optional): Dimensions of the input data (channels, height, width). Defaults to (1, 28, 28).
            conv_param_1 (_type_, optional): Parameters for the first convolutional layer. Defaults to {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}.
            conv_param_2 (_type_, optional): Parameters for the second convolutional layer. Defaults to {'filter_num': 16, 'filter_size': 3, 'pad': 1, 'stride': 1}.
            conv_param_3 (_type_, optional): Parameters for the third convolutional layer. Defaults to {'filter_num': 32, 'filter_size': 3, 'pad': 1, 'stride': 1}.
            conv_param_4 (_type_, optional): Parameters for the fourth convolutional layer. Defaults to {'filter_num': 32, 'filter_size': 3, 'pad': 2, 'stride': 1}.
            conv_param_5 (_type_, optional): Parameters for the fifth convolutional layer. Defaults to {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}.
            conv_param_6 (_type_, optional): Parameters for the sixth convolutioanl layer. Defaults to {'filter_num': 64, 'filter_size': 3, 'pad': 1, 'stride': 1}.
            hidden_size (int, optional): Size of the hidden layer. Defaults to 50.
            output_size (int, optional): Size of the output layer. Defaults to 10.
        """
        pre_node_nums = np.array([1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값
        
        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = wight_init_scales[idx] * np.random.randn(conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = wight_init_scales[6] * np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = wight_init_scales[7] * np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                                       conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                                       conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                                       conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                                       conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                                       conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                                       conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x: np.ndarray, train_flg: bool=False) -> np.ndarray:
        """Perform the forward pass to make predictions.

        Args:
            x (numpy.ndarray): Input data.
            train_flg (bool, optional): Flag indicating training mode. Defaults to False.

        Returns:
            numpy.ndarray: Predicted probabilities.
        """
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
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
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)
    
    def accuracy(self, x: np.ndarray, t: np.ndarray, batch_size: int=100) -> float:
        """Calculate the accuracy for a given input and target.

        Args:
            x (numpy.ndarray): Input data.
            t (numpy.ndarray): Target labels.
            batch_size (int, optional): Batch size for accuracy calculation. Defaults to 100.

        Returns:
            float: Accuracy value.
        """
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]
    
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

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads
    
    def save_params(self, file_name: str="params.pkl"):
        """Save the model parameters to a file.

        Args:
            file_name (str, optional): File name to save the paramters. Defaults to "params.pkl".
        """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name: str="params.pkl"):
        """Load the model parameters from a file.

        Args:
            file_name (str, optional): File name to load the parameters. Defaults to "params.pkl".
        """
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]