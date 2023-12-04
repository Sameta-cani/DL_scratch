import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x: np.array) -> np.array:
    """Compute the sigmoid function.

    Args:
        x (np.array): Input array.

    Returns:
        np.array: Output array of the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Sigmoid Function")
plt.grid(True)
plt.show()