# coding: utf-8
import numpy as np

def numerical_gradient_1d(f: callable, x: np.ndarray) -> np.ndarray:
    """Compute the numerical gradient of a 1-dimensional function.

    Args:
        f (callable): The function for which the gradient is calculated.
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The numerical gradient of the function at the given input.
    """
    h = 1e-4 # Step size for numerical differentiation
    grad = np.zeros_like(x)
    
    for idx, tmp_val in enumerate(x):
        # Calculate f(x + h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # Calculate f(x - h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        # Numerical gradient computation
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        # Restore the original value of x[idx]
        x[idx] = tmp_val

    return grad


def numerical_gradient_2d(f: callable, X: np.ndarray) -> np.ndarray:
    """Compute the numerical gradient of a 2-dimensional function.

    Args:
        f (callable): The function for which the gradient is calculated.
        X (np.ndarray): The input array. If 1-dimensional, computes 1D numerical gradient.

    Returns:
        np.ndarray: The numerical gradient of the function at the given input.
    """
    if X.ndim == 1:
        return numerical_gradient_1d(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_1d(f, x)
        
        return grad


def numerical_gradient(f: callable, x: np.ndarray) -> np.ndarray:
    """Compute the numerical gradient of a function.

    Args:
        f (callable): The function for which the gradient is calculated.
        x (np.ndarray): The input array.

    Returns:
        np.ndarray: The numerical gradient of the function at the given input.
    """
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx, tmp_val in np.ndenumerate(x):
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val

    return grad