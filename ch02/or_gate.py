import numpy as np

def OR(x1: int, x2: int) -> int:
    """Implementation of the OR gate using a perceptron.

    Args:
        x1 (int): Input 1 (0 or 1)
        x2 (int): Input 2 (0 or 1)

    Returns:
        int: Output of the OR gate (0 or 1)
    """
    x = np.array([x1, x2])
    weights = np.array([0.5, 0.5])
    bias = -0.2
    result = np.dot(x, weights) + bias
    return int(result > 0)

if __name__ == '__main__':
    test_cases = [(0, 0), (1, 0), (0, 1), (1, 1)]
    for xs in test_cases:
        y = OR(xs[0], xs[1])
        print(f"{xs} -> {y}")