import numpy as np
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.util import im2col, col2im

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

class Selu:
    """Scaled Exponential Linear Unit (SeLU) activation function for neural networks.

    This class represents the SeLU activation function used in neural networks.
    It has forward and backward methods to perform forward and backward passes.

    Attributes:
        alpha (float): Slope of the negative region.
        scale (float): Scaling factor for the output.
        mask (numpy.ndarray): A boolean mask indicating the locations where the input is less than or equal to 0.
    """
    def __init__(self, alpha=1.67326, scale=1.0507):
        self.alpha = alpha
        self.scale = scale
        self.mask = None
        self.input_data = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass and return the result of SeLU activation.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The result of the SeLU activation.
        """
        self.mask = (x <= 0)
        self.input_data = x
        out = self.scale * np.where(x > 0, x, self.alpha * (np.exp(np.clip(x, -700, 700)) - 1))
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform backward pass and return the gradient with respect to the input.

        Args:
            dout (numpy.ndarray): The gradient from the next layer.

        Returns:
            numpy.ndarray: The gradient with respect to the input.
        """
        dx_pos = dout * self.scale * np.where(self.input_data > 0, 1, 0)
        dx_neg = dout * self.scale * self.alpha * np.exp(np.clip(self.input_data, -700, 700)) * np.where(self.input_data <= 0, 1, 0)
        dx = dx_pos + dx_neg
        return dx


class ELU:
    """Exponential Linear Unit (ELU) activation function for neural networks.

    This class represents the ELU activation function used in neural networks.
    It has forward and backward methods to perform forward and backward passes.

    Attributes:
        alpha (float): The alpha parameter controlling the slope of the negative region.
        input (numpy.ndarray): The input data saved during the fowrad pass.
    """
    def __init__(self, alpha: float=1.0):
        """Initializes the ELU activation function with the given alpha parameter.

        Args:
            alpha (float, optional): The alpha parameter controlling the slope of negative region. Defaults to 1.0.
        """
        self.alpha = alpha
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform forward pass and return the result of ELU activation.

        Args:
            x (numpy.ndarray): The input array.

        Returns:
            numpy.ndarray: The result of the ELU activation.
        """
        self.input = x
        out = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform backward pass and return the gradient with respect to the input.

        Args:
            dout (numpy.ndarray): The gradient from the next layer.

        Returns:
            numpy.ndarray: The gradient with respect to the input.
        """
        dx = dout * np.where(self.input > 0, 1, self.alpha * np.exp(self.input))
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
    
class Dropout:
    """
    Dropout layer for regularization during training.

    References:
        - Dropout: A Simple Way to Prevent Neural Networks from Overfitting
          (http://jmlr.org/papers/volume15/srivastava14a.old/srivastava14a.pdf)
    
    Args:
        dropout_ratio (float, optional): Dropout ratio, the probability of setting a neuron to zero during training.
            Defaults to 0.5.
    """
    def __init__(self, dropout_ratio: float = 0.5):
        """
        Initialize the Dropout layer.

        Attributes:
            dropout_ratio (float): Dropout ratio, the probability of setting a neuron to zero during training.
            mask (np.array): Dropout mask used during the forward pass.
        """
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x: np.array, train_flg: bool = True) -> np.array:
        """
        Perform the forward pass of the Dropout layer.

        Args:
            x (np.array): Input data.
            train_flg (bool, optional): Flag indicating whether the model is in training mode.
                Defaults to True.

        Returns:
            np.array: Output data after applying Dropout.
        """
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout: np.array) -> np.array:
        """
        Perform the backward pass of the Dropout layer.

        Args:
            dout (np.array): Gradient of the loss with respect to the output.

        Returns:
            np.array: Gradient of the loss with respect to the input.
        """
        return dout * self.mask

class AlphaDropout:
    """Alpha Dropout layer for neural networks.

    Alpha Dropout is a type of dropout regularization that maintains the mean
    and variance of the activations close to the original values during training,
    which can be beneficial for certain types of networks.

    Args:
        dropout_ratio (float, optional): The dropout ratio, indicating the fraction
            of input units to drop. Defaults to 0.5.
    """
    def __init__(self, dropout_ratio: float = 0.5):
        """Initialize AlphaDropout with the given dropout ratio."""
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x: np.ndarray, train_flg: bool = True) -> np.ndarray:
        """Apply Alpha Dropout during the forward pass."""
        if train_flg:
            self.mask = (np.random.rand(*x.shape) > self.dropout_ratio)
            return x * self.mask * (1 - self.dropout_ratio)
        else:
            return x * (1 - self.dropout_ratio)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform the backward pass for Alpha Dropout."""
        return (dout / (1 - self.dropout_ratio)) * self.mask


class BatchNormalization:
    """
    Batch Normalization layer for neural networks.

    References:
        - Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
          (http://arxiv.org/abs/1502.03167)
    
    Args:
        gamma (np.array): Scale parameter.
        beta (np.array): Shift parameter.
        momentum (float, optional): Momentum for the moving average. Defaults to 0.9.
        running_mean (np.array, optional): Running mean for inference. Defaults to None.
        running_var (np.array, optional): Running variance for inference. Defaults to None.
    """
    def __init__(self, gamma: np.array, beta: np.array, momentum: float = 0.9, running_mean: np.array = None, running_var: np.array = None):
        """
        Initialize the BatchNormalization layer.

        Attributes:
            gamma (np.array): Scale parameter.
            beta (np.array): Shift parameter.
            momentum (float): Momentum for the moving average.
            input_shape (tuple): Shape of the input data.
            running_mean (np.array): Running mean for inference.
            running_var (np.array): Running variance for inference.
            batch_size (int): Size of the current mini-batch.
            xc (np.array): Centered input data.
            xn (np.array): Normalized input data.
            std (np.array): Standard deviation of the input data.
            dgamma (np.array): Gradient of the loss with respect to gamma.
            dbeta (np.array): Gradient of the loss with respect to beta.
        """
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None

        # Running statistics for inference
        self.running_mean = running_mean
        self.running_var = running_var  

        # Intermediate data for backward pass
        self.batch_size = None
        self.xc = None
        self.xn = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x: np.array, train_flg: bool = True) -> np.array:
        """
        Perform the forward pass of the BatchNormalization layer.

        Args:
            x (np.array): Input data.
            train_flg (bool, optional): Flag indicating whether the model is in training mode.
                Defaults to True.

        Returns:
            np.array: Output data after applying BatchNormalization.
        """
        self.input_shape = x.shape
        if x.ndim != 2:
            x = x.reshape(len(x), -1)

        out = self.__forward(x, train_flg)
        
        return out.reshape(*self.input_shape)
            
    def __forward(self, x: np.array, train_flg: bool) -> np.array:
        """
        Perform the forward pass of the BatchNormalization layer.

        Args:
            x (np.array): Input data.
            train_flg (bool): Flag indicating whether the model is in training mode.

        Returns:
            np.array: Output data after applying BatchNormalization.
        """
        if self.running_mean is None:
            self.running_mean = np.zeros(x.shape[1])
            self.running_var = np.zeros(x.shape[1])
                        
        if train_flg:
            mu = np.mean(x, axis=0)
            xc = x - mu
            var = np.mean(xc**2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std
            
            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var            
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))
            
        out = self.gamma * xn + self.beta 
        return out

    def backward(self, dout: np.array) -> np.array:
        """
        Perform the backward pass of the BatchNormalization layer.

        Args:
            dout (np.array): Gradient of the loss with respect to the output.

        Returns:
            np.array: Gradient of the loss with respect to the input.
        """
        if dout.ndim != 2:
            dout = dout.reshape(len(dout), -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout: np.array) -> np.array:
        """
        Perform the backward pass of the BatchNormalization layer.

        Args:
            dout (np.array): Gradient of the loss with respect to the output.

        Returns:
            np.array: Gradient of the loss with respect to the input.
        """
        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size
        
        self.dgamma = dgamma
        self.dbeta = dbeta
        
        return dx
    

class Convolution:
    def __init__(self, W: np.ndarray, b: np.ndarray, stride: int=1, pad: int=0):
        """Initialize a convolutional layer with learnable parameters.

        Args:
            W (numpy.ndarray): Weight parameters for the convolutional layer of shape (FN, C, FH, FW),
            where FN is the number of filters, C is the number of channels,
            FH is the filter height, and FW is the filter width.
            b (numpy.ndarray): Bias parameters for the convolutional layer of shape (FN,).
            stride (int, optional): Stride value for filter movement. Defaults to 1.
            pad (int, optional): Padding value for input data. Defaults to 0.
        """
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # Intermediate data (using during backward pass)
        self.x = None
        self.col = None
        self.col_W = None

        # Gradients for weights and biases
        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the convolutional layer.

        Args:
            x (numpy.ndarray): Input data of shape (N, C, H, W), where N is the batch size,
                               C is the number of channels, H is the height, and W is the width.

        Returns:
            numpy.ndarray: Output of the convolutional layer after the forward pass.
        """
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perforn the backward pass of the convolutional layer.

        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx
    
class Pooling:
    def __init__(self, pool_h: int, pool_w: int, stride: int=1, pad: int=0):
        """Initialize a pooling layer.

        Args:
            pool_h (int): Height of the pooling window.
            pool_w (int): Width of the pooling window.
            stride (int, optional): Stride value for pooling. Defaults to 1.
            pad (int, optional): Padding value for input data. Defaults to 0.
        """
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Perform the forward pass of the pooling layer.

        Args:
            x (numpy.ndarray): Input data of shape (N, C, H, W), where N is the batch size,
                            C is the number of channels, H is the height, and W is the width.

        Returns:
            numpy.ndarray: Output of the pooling layer after the forward pass.
        """
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out
    
    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Perform the backward pass of the pooling layer.

        Args:
            dout (numpy.ndarray): Gradient of the loss with respect to the output of this layer.

        Returns:
            numpy.ndarray: Gradient of the loss with respect to the input of this layer.
        """
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx