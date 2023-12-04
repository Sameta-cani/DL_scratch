import numpy as np
import matplotlib.pyplot as plt

def relu(x: np.array) -> np.array:
    """Compute the Redctified Linear Unit (ReLU) function.

    Args:
        x (np.array): Input array.

    Returns:
        np.array: Output array of the ReLU function.
    """
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.ylim(-1.0, 5.5)
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('ReLU Function')
plt.grid(True)
plt.show()