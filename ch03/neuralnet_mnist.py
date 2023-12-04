import sys
import os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    """Load the MNIST test data and labels."""
    return load_mnist(normalize=True, flatten=True, one_hot_label=False)[1]

def init_network():
    """Load the pre-trained neural network parameters."""
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    """Predict the label for the input data using the neural network."""
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    # Feedforward calculation
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# Load data and network
x_test, t_test = get_data()
network = init_network()

# Make predictions and evaluate accuracy
accuracy_cnt = np.sum([np.argmax(predict(network, x)) == t for x, t in zip(x_test, t_test)])
accuracy = accuracy_cnt / len(x_test)

print(f"Accuracy: {accuracy}")