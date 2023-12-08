import numpy as np

class SGD:
    """Stochastic Gradient Descent (SGD) optimizer.

    Parameters:
        lr (float, optional): Learning rate for the optimization. Defaults to 0.01.

    Attributes:
        lr (float): The learning rate for the optimization.    
    """

    def __init__(self, lr: float=0.01):
        """Initialize the SGD optimizer with a given learning rate.

        Args:
            lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
        """
        self.lr = lr

    def update(self, params: dict, grads: dict):
        """Update the parameters using the SGD optimization algorithm.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to each parameter.
        """
        for key, value in params.items():
            params[key] -= self.lr * grads[key]

class Momentum:
    """Momentum optimizer.

    Parameters:
        lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
        momentum (float, optional): Momentum factor. Defaults to 0.9.    
    """
    def __init__(self, lr: float=0.01, momentum: float=0.9):
        """Initialize the Momentum optimizer with a given learning rate and momentum.

        Args:
            lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
            momentum (float, optional): Momentum factor. Defaults to 0.9.
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params: dict, grads: dict):
        """Update the parameters using the Momentum optimization algorithm.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to each parameter.
        """
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        for key, val in params.items():
            v = self.v[key]
            v = self.momentum * v - self.lr * grads[key]
            params[key] += v
            self.v[key] = v

class Nesterov:
    """Nesterov Accelerated Gradient (NAG) optimizer.

    Parameters:
        lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
        momentum (float, optional): Momentum factor. Defaults to 0.9.
    """

    def __init__(self, lr: float=0.01, momentum: float=0.9):
        """Initialize the Nesterov optimizer with a given learning rate and momentum.

        Args:
            lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
            momentum (float, optional): Momentum factor. Defaults to 0.9.
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params: dict, grads: dict):
        """Update the parameters using the Nesterov Accelerated Gradient (NAG) optimization.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to each parameter.
        """
        if self.v is None:
            self.v = {key: np.zeros_like(val) for key, val in params.items()}

        for key, val in params.items():
            v = self.v[key]
            v *= self.momentum
            v -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * v - (1 + self.momentum) * self.lr * grads[key]
            self.v[key] = v

class AdaGrad:
    """AdaGrad optimizer.

    Parameters:
        lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
    """
    def __init__(self, lr: float=0.01):
        """Initialize the AdaGrad optimizer with a given learning rate.

        Args:
            lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
        """
        self.lr = lr
        self.h = None

    def update(self, params: dict, grads: dict):
        """Update the parameters using the AdaGrad optimization algorithm.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to each parameter.
        """
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}

        for key, val in params.items():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class RMSprop:
    """RMSprop optimizer.

    Parameters:
        lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
        decay_rate (float, optional): Decay rate for the moving average of squared gradients. Defaults to 0.99.
    """

    def __init__(self, lr: float=0.01, decay_rate: float=0.99):
        """Initialize the RMSprop optimizer with a given learning rate and decay rate.

        Args:
            lr (float, optional): Learning rate for the optimization. Defaults to 0.01.
            decay_rate (float, optional): Decay rate for the moving average of squared gradients. Defaults to 0.99.
        """
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params: dict, grads: dict):
        """Update the parameters using the RMSprop optimization algorithm.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to each parameter.
        """
        if self.h is None:
            self.h = {key: np.zeros_like(val) for key, val in params.items()}

        for key, val in params.items():
            h = self.h[key]
            h *= self.decay_rate
            h += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(h) + 1e-7)
            self.h[key] = h

class Adam:
    """Adam optimizer.

    Parameters:
        lr (float, optional): Learning rate for the optimization. Defaults to 0.001.
        beta1 (float, optional): Exponential decay rate for the first momentum estimate. Defaults to 0.9.
        beta2 (float, optional): Exponential decay rate for the second momentum estimate. Defaults to 0.999.
    """

    def __init__(self, lr: float=0.001, beta1: float=0.9, beta2: float=0.999):
        """Initialize the Adam optimizer with a given learning rate, beta1, and beta2.

        Args:
            lr (float, optional): Learning rate for the optimization. Defaults to 0.001.
            beta1 (float, optional): Exponential decay rate for the first momentum estimate. Defaults to 0.9.
            beta2 (float, optional): Expoential decay rate for the second momentum estimate. Defaults to 0.999.
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params: dict, grads: dict):
        """Update the parameters using the Adam optimization algorithm.

        Args:
            params (dict): Dictionary of parameters to be updated.
            grads (dict): Dictionary of gradients corresponding to each parameter.
        """
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)

        for key in params.keys():
            m, v, grad = self.m[key], self.v[key], grads[key]
            m += (1 - self.beta1) * (grad - m)
            v += (1 - self.beta2) * (grad**2 - v)
            params[key] -= lr_t * m / (np.sqrt(v) + 1e-7)
            self.m[key], self.v[key] = m, v