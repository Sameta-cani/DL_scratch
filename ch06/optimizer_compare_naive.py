import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from common.optimizer import *

def f(x: float, y: float) -> float:
    """Objective function for optimzation.

    Args:
        x (float): Input variable x.
        y (float): Input variable y.

    Returns:
        float: The result of the objective function.
    """
    return x**2 / 20.0 + y**2

def df(x: float, y: float) -> tuple:
    """Gradient of the objective function.

    Args:
        x (float): Input variable x.
        y (float): Input variable y.

    Returns:
        tuple: Gradient with respect to x and y.
    """
    return x / 10.0, 2.0 * y

def plot_optimization(optimizer: object, init_pos: tuple, title: str):
    """Plot the optimization path for a given optimizer.

    Args:
        optimizer (object): The optimization algorithm to be visualized.
        init_pos (tuple): Initial position for optimization (x, y).
        title (str): Title for the plot.
    """
    params = {'x': init_pos[0], 'y': init_pos[1]}
    grads = {'x': 0, 'y': 0}

    x_history, y_history = [params['x']], [params['y']]

    for _ in range(30):
        grads['x'], grads['y'] = df(params['x'], params['y'])
        optimizer.update(params, grads)
        x_history.append(params['x'])
        y_history.append(params['y'])

    x = np.arange(-10, 10, 0.01)
    y = np.arange(-10, 10, 0.01)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)
    mask = Z > 7
    Z[mask] = 0

    plt.plot(x_history, y_history, 'o-', label=title)
    plt.contour(X, Y, Z, colors='gray', linestyles='dashed')
    plt.scatter(0, 0, marker='x', color='black')
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")

def main():
    """Main function to visualize optimization paths for various optimizers.
    """
    optimizers = OrderedDict()
    optimizers["SGD"] = SGD(lr=0.95)
    optimizers["Momentum"] = Momentum(lr=0.1)
    optimizers["Nesterov"] = Nesterov(lr=0.1)
    optimizers["AdaGrad"] = AdaGrad(lr=1.5)
    optimizers["RMSprop"] = RMSprop(lr=0.1)
    optimizers["Adam"] = Adam(lr=0.3)

    plt.figure(figsize=(15, 10))

    for idx, (key, optimizer) in enumerate(optimizers.items()):
        plt.subplot(2, 3, idx+1)
        plot_optimization(optimizer, init_pos=(-7.0, 2.0), title=key)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
