import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from deep_convnet import DeepConvNet
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

network = DeepConvNet()
network.load_params("deep_convnet_params.pkl")

print("calculating test accuracy ... ")

classified_ids = []
acc = 0.0
batch_size = 100

for i in range(0, x_test.shape[0], batch_size):
    tx_batch, tt_batch = x_test[i:i+batch_size], t_test[i:i+batch_size]
    y_batch = network.predict(tx_batch, train_flg=False)
    classified_ids.append(np.argmax(y_batch, axis=1))
    acc += np.sum(np.argmax(y_batch, axis=1) == tt_batch)

acc /= x_test.shape[0]
print(f"Test accuracy: {acc:.4f}")

classified_ids = np.concatenate(classified_ids)

max_view = 19
current_view = 0

fig, axes = plt.subplots(4, 5, figsize=(10, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.2, wspace=0.2)

mis_pairs = {}

for i, (is_correct, img, label, inferred_label) in enumerate(zip(classified_ids == t_test, x_test, t_test, classified_ids)):
    if not is_correct:
        ax = axes[current_view // 5, current_view % 5]
        ax.imshow(img.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
        ax.set_title(f"Label: {label}\nInferred: {inferred_label}")
        ax.set_xticks([])
        ax.set_yticks([])
        mis_pairs[current_view] = (label, inferred_label)
        current_view += 1

        if current_view > max_view:
            break

print("======= misclassified result =======")
print("{view index: (label, inference), ...}")
print(mis_pairs)

plt.show()