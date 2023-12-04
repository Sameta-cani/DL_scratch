import numpy as np
import matplotlib.pyplot as plt

def step_function(x: np.array) -> np.array:
    """Implement the step function.

    Args:
        x (np.array): Input array.

    Returns:
        np.array: Output array of the step function (0 or 1).
    """
    return np.array(x > 0, dtype=np.int32)

X = np.arange(-5.0, 5.0, 0.1)
Y = step_function(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.xlabel("Input")
plt.ylabel("Output")
plt.title("Step Function")
plt.grid(True)
plt.show()