import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def initialize_weights(node_num: int, initialization_method: str='xavier') -> np.ndarray:
    """Initialize weights for a neural network layer.

    Args:
        node_num (int): Number of nodes in the layer.
        initialization_method (str, optional): Method for weight initialization. Defaults to 'xavier'.
            - 'none': Initialize weights with zeros.
            - 'normal': Initialize weights with random values from a normal distribution with mean 0 and standard deviation 1.
            - 'xavier': Initialize weights with random values from a normal distribution with mean 0 and stadnard deviation 1/number_of_input_nodes.
            - 'he': Initialize weights with random values from a normal distribution with mean 0 and standard deviation 2/number_of_input_nodes.

    Returns:
        numpy.ndarray: Initialized weights matrix.
    """
    if initialization_method == 'none':
        return np.zeros((node_num, node_num))
    elif initialization_method == 'normal':
        return np.random.randn(node_num, node_num) * 1
    elif initialization_method == 'xavier':
        return np.random.randn(node_num, node_num) * np.sqrt(1.0 / node_num)
    elif initialization_method == 'he':
        return np.random.randn(node_num, node_num) * np.sqrt(2.0 / node_num)

def plot_activations(activations: dict) -> None:
    """Plot histograms of activation values for each layer in a nueral network.

    Args:
        activations (dict): Dictionary containing layer indices as keys and activation arrays as values.
    """
    for i, a in activations.items():
        plt.subplot(1, len(activations), i + 1)
        plt.title(f'{str(i + 1)}-layer')
        if i != 0:
            plt.yticks([], [])
        plt.hist(a.flatten(), 30, range=(0, 1))
    plt.show()

def main():
    input_data = np.random.randn(1000, 100)
    node_num = 100
    hidden_layer_size = 5
    activations = {}

    x = input_data

    for i in range(hidden_layer_size):
        if i != 0:
            x = activations[i - 1]

        w = initialize_weights(node_num, initialization_method='normal')
        a = np.dot(x, w)

        z = sigmoid(a)
        # z = relu(a)
        # z = tanh(a)

        activations[i] = z

    plot_activations(activations)

if __name__ == "__main__":
    main()