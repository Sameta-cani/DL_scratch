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

def im2col(input_data: np.ndarray, filter_h: int, filter_w: int, stride: int=1, pad: int=0) -> np.ndarray:
    """Perform the image to column operation on the input data.

    Args:
        input_data (numpy.ndarray): Input image data of shape (N, C, H, W), where N is the batch size,
                                    C is the number of channels, H is the height, and W is the wdith.
        filter_h (int): Height of the filter (kernel).
        filter_w (int): Width of the filter (kernel).
        stride (int, optional): Stride value for filter movement. Defaults to 1.
        pad (int, optional): Padding value for input data. Defaults to 0.

    Returns:
        numpy.ndarray: 2D array representing the im2col result with shape (N * out_h * out_w, C * filter_h * filter_w),
                       where out_h and out_w are determined by the input dimensions, filter size, stride, and padding.
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # Apply padding
    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col

def col2im(col: np.ndarray, input_shape: tuple, filter_h: int, filter_w: int, stride: int=1, pad: int=0) -> np.ndarray:
    """Perform the col2im operation to reconstruct the original image from the im2col result.

    Args:
        col (numpy.ndarray): 2D array representing the im2col result.
        input_shape (tuple): Shape of the original input image (N, C, H, W), where N is the batch size,
                             C is the number of channels, H is the height, and  W is the width.
        filter_h (int): Height of the filter (kernel).
        filter_w (int): Width of the filter (kernel).
        stride (int, optional): Stride value for filter movement. Defaults to 1.
        pad (int, optional): Padding value for input data. Defaults to 0.

    Returns:
        numpy.ndarray: Reconstructed image width shape (N, C, H', W'), where H' and W' are determined by the input
                       dimensions, filter size, stride, and padding. The result has been adjusted to remove padding.
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h) // stride + 1
    out_w = (W + 2*pad - filter_w) // stride + 1

    # Reshape and transpose col array
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    # Initialize output image with zeros
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

    # Perform the col2im operation
    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    # Extract the region without padding
    img = img[:, :, pad:H + pad, pad:W + pad]
    
    return img

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