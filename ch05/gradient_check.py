import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# Load MNIST data
(x_train, t_train), (_, _) = load_mnist(normalize=True, one_hot_label=True)

# Create a TwoLayerNet instance
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# Take a small batch for testing
x_batch = x_train[:3]
t_batch = t_train[:3]

# Calculate numerical gradients and backpropagation gradients
grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

# Iterate over keys and compare gradients
for key, grad_num, grad_bp in zip(grad_numerical.keys(), grad_numerical.values(), grad_backprop.values()):
    diff = np.average(np.abs(grad_bp - grad_num))
    print(f"{key}: {diff:.10f}")