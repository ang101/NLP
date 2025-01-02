import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Attention

def detect_and_set_device():
    """Detects if a GPU is available and sets the device accordingly.

    Returns:
        str: 'GPU' if a GPU is available and memory growth is set successfully,
            otherwise 'CPU'.
    """
    return 'GPU' if tf.test.is_gpu_available() else 'CPU'

def build_LSTM_model(vocab_size, input_length):
    """Build LSTM model"""
    model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size + 1, output_dim=128, input_length=input_length),  # +1 for <UNK>
    tf.keras.layers.LSTM(128, return_sequences=False),
    tf.keras.layers.Dense(vocab_size + 1, activation='softmax')  # +1 for <UNK>
])
    return model


def compile_LSTM_model(model):
    """Compile the model."""
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

def train_LSTM_model(model, X_train, y_train, X_test, y_test, epochs=3, batch_size=64):
    """Train the model."""
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test)
    )

def create_input_output(sequences, sequence_length=2):
    """Prepare input-output data."""
    X, y = [], []
    for i in range(len(sequences) - sequence_length):
        X.append(sequences[i:i + sequence_length - 1])
        y.append(sequences[i + sequence_length - 1])
    return np.array(X), np.array(y)
