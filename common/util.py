import numpy as np
import matplotlib.pyplot as plt

def smooth_curve(x):
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

def shuffle_dataset(x, t):
    permutation = np.random.permutation(x.shape[0])
    x = x[permutation, :] if x.ndim == 2 else x[permutation, :, :, :]
    t = t[permutation]

    return x, t

def plot_image(i, predictions_array, true_label, images, class_names):
    """
    Display a single image along with its predicted and true labels.

    Args:
        i (int): Index of the image in the dataset.
        predictions_array (numpy.ndarray): Array of prediction probabilities for each class.
        true_label (int): True label of the image.
        images (numpy.ndarray): Array of image data.
        class_names (list): List of class names.

    Returns:
        None

    Display an image without grid and axes, and show its predicted label, confidence percentage,
    and true label in the xlabel. The predicted label is displayed in blue if correct,
    and in red if incorrect.

    Example:
    plot_image(0, predictions_array, true_label, images, CLASS_NAMES)
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], images[i]
  
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img.reshape((28, 28)), cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    color = 'blue' if predicted_label == true_label else 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                           100 * np.max(predictions_array),
                                           class_names[true_label]),
               color=color)


def plot_value_array(i: int, predictions_array: np.ndarray, true_label: int):
  """Display a bar plot of prediction probabilities along with color-coded bars for the predicted and true labels.

  Args:
      i (int): Index of the image in the dataset.
      predictions_array (numpy.ndarray): Array of prediction probabilities for each class.
      true_label (int): True label of the image.

  Returns:
    None

  Display a bar plot without grid and ticks, with color-coded bars for the predicted and true labels.

  Example:
  plot_value_array(0, predictions_array, true_label)
  """
  predictions_array, true_label = predictions_array[i], true_label[i]

  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.bar(range(10), predictions_array, color='#777777')
  plt.ylim([0, 1])

  predicted_label = np.argmax(predictions_array)

  plt.bar(predicted_label, predictions_array[predicted_label], color='red')
  plt.bar(true_label, predictions_array[true_label], color='blue')
  
def plot_image_grid(X_test, y_test, y_hat, class_names, num_rows=5, num_cols=3):
    """
    Display a grid of images along with their predicted and true labels and probability arrays.

    Args:
        X_test (numpy.ndarray): Array of image data.
        y_test (numpy.ndarray): True labels of the images.
        y_hat (numpy.ndarray): Predicted labels with probability arrays for each class.
        class_names (list): List of class names.
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.

    Returns:
        None

    Display a grid of images with their predicted and true labels on the left side
    and the probability arrays on the right side.

    Example:
    plot_image_grid(X_test, y_test, y_hat, CLASS_NAMES, num_rows=5, num_cols=3)
    """
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))

    for i in range(num_images):
        # Plot Image
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
        plot_image(i, y_hat, y_test, X_test, class_names)

        # Plot Value Array
        plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
        plot_value_array(i, y_hat, y_test)