import numpy as np
import matplotlib.pyplot as plt

from sigmoid import sigmoid
from step_function import step_function

x = np.arange(-5.0, 5.0, 0.1)
y_sigmoid = sigmoid(x)
y_step = step_function(x)

# Plot the sigmoid function with a blue solid line.
plt.plot(x, y_sigmoid, label='Sigmoid', color='blue')
# Plot the step function with a black dashed line.
plt.plot(x, y_step, label='Step', color='black', linestyle='--')

plt.ylim(-0.1, 1.1)
plt.legend()  # Display the legend.
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Comparison of Sigmoid and Step Functions')
plt.grid(True)
plt.show()