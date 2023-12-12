import numpy as np



def identity_function(x: np.ndarray) -> np.ndarray:
    """Return the input array as is.

    Args:
        x (np.ndarray): The input array._

    Returns:
        np.ndarray: The input array itself.
    """
    return x


def step_function(x: np.ndarray) -> np.ndarray:
    """Return the step function output for the input array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The step function output (0 or 1) based on the input array's elements.
    """
    return (x > 0).astype(np.int32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function for the input array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The sigmoid function output for each element of the input array.
    """
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """Compute the gradient of the sigmoid function.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The gradient of the sigmoid function for each element of the input array.
    """
    sig = sigmoid(x)
    return (1.0 - sig) * sig


def relu(x: np.ndarray) -> np.ndarray:
    """Compute the Rectified Linear Unit (ReLU) for the input array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The ReLU output for each element of the input array.
    """
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Compute the gradient of the Rectified Linear Unit (ReLU) function.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The gradient of the ReLU function for each element of the input array.
    """
    return np.where(x >= 0, 1, 0)

def selu(x, alpha=1.67326, scale=1.0507):
    return scale * np.where(x > 0, x, alpha * (np.exp(x) - 1))


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute the softmax function for the input array.

    Args:
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The softmax output for each element of the input array.
    """
    if x.ndim == 2:
        x = x - np.max(x, axis=1, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        return y
    
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def mean_squared_error(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute the mean squared error between predicted and target arrays.

    Args:
        y (np.ndarray): The predicted array.
        t (np.ndarray): The target array.

    Returns:
        np.ndarray: The mean squared error between y and t.
    """
    return 0.5 * np.sum((y - t)**2)


def cross_entropy_error(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute the cross-entropy error between predicted and target arrays.

    Args:
        y (np.ndarray): The predicted array.
        t (np.ndarray): The target array.

    Returns:
        np.ndarray: The cross-entropy error between y and t.
    """
    if y.ndim == 1:
        t = t.reshape(1, -1)
        y = y.reshape(1, -1)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def softmax_loss(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute the softmax loss between predicted and target arrays.

    Args:
        X (np.ndarray): The input array.
        t (np.ndarray): The target array.

    Returns:
        np.ndarray: The softmax loss between X and t.
    """
    return cross_entropy_error(softmax(X), t)