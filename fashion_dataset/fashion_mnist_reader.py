import os
import gzip
import numpy as np

import numpy as np

def change_one_hot_label(X: np.array) -> np.array:
    """Converts an array of labels to one-hot encoded format.

    Args:
        X (np.array): An array containing integer labels.

    Returns:
        np.array: A 2D numpy array representing the one-hot encoded labels.
    """
    num_classes = 10
    
    # Use numpy's eye function to create a one-hot encoded matrix
    T = np.eye(num_classes, dtype=int)[X]

    return T

def load_mnist(path: str = '/content', kind: str = 'train', normalize: bool = True, one_hot_label: bool = False) -> tuple:
    """Load MNIST data from specified path.

    Args:
        path (str, optional): Path to the MNIST data files. Defaults to '/content'.
        kind (str, optional): 'train' or 't10k'. Defaults to 'train'.
        normalize (bool, optional): Whether to normalize pixel values. Defaults to True.
        one_hot_label (bool, optional): Whether to use one-hot encoding for labels. Defaults to False.

    Returns:
        tuple: A tuple containing the loaded images and labels.
    """
    # Construct file paths
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    # Read label data from the gzip file
    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    # Read image data from the gzip file and reshape
    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)

    # Normalize pixel values if specified
    if normalize:
        images = images.astype(np.float32)
        images = (images - images.mean()) / images.std()
    
    # Convert labels to one-hot encoding if specified
    if one_hot_label:
        labels = change_one_hot_label(labels)

    return images, labels
