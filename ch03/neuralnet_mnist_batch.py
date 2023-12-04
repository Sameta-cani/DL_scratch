import sys, os
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
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']


    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x_test, t_test = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x_test), batch_size):
    x_batch, t_batch = x_test[i:i+batch_size], t_test[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p_batch = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p_batch == t_batch)

accuracy = accuracy_cnt / len(x_test)
print(f"Accuracy: {accuracy}")