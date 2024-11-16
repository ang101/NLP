import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Setup
def detect_and_set_device():
    """Detects if a GPU is available and sets the device accordingly.

    Returns:
        str: 'GPU' if a GPU is available and memory growth is set successfully,
            otherwise 'CPU'.
    """
    return 'GPU' if tf.test.is_gpu_available() else 'CPU'

def display_samples(images, labels, num_samples=16):
    plt.figure(figsize=(8, 8))
    for i in range(num_samples):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def plot_metrics(history):
  # Plotting training & validation accuracy values
  plt.figure(figsize=(12, 4))

  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Train Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(loc='upper left')

  # Plotting training & validation loss values
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Train Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(loc='upper left')

  plt.tight_layout()
  plt.show()

def plot_predictions(x_test, y_true, y_pred, num_samples=10):
  """Plots a grid of images with true and predicted labels, color-coded based on accuracy.

  Args:
    x_test: Test images.
    y_true: True labels.
    y_pred: Predicted labels.
    num_samples: Number of samples to plot.
  """

  class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

  num_plots = min(num_samples, len(y_true))
  fig, axes = plt.subplots(nrows=int(np.ceil(np.sqrt(num_plots))), ncols=int(np.ceil(np.sqrt(num_plots))), figsize=(15, 15))

  for i in range(num_plots):
    ax = axes.flatten()[i]
    ax.imshow(x_test[i], cmap='gray')

    # Determine the color based on accuracy
    if y_true[i] == y_pred[i]:
      color = 'green'
    else:
      color = 'red'

    # Set the labels with the appropriate color and adjust padding
    ax.set_xlabel(f"True: {class_names[y_true[i]]} \n Predicted: {class_names[y_pred[i]]}", color=color, fontsize=12, ha='center', va='top')

    # Adjust subplot spacing
    plt.subplots_adjust(hspace=0.3)  # Adjust vertical spacing as needed

  plt.show()