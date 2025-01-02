import pandas as pd
from create_mappings import create_mappings

def test_data(data):
    """
    Test if the shape of the dataset is (1048575, 4).

    Args:
        data (pd.DataFrame): The dataset to test.

    Raises:
        AssertionError: If the dataset shape is not (1048575, 4).
    """
    expected_shape = (1048575, 4)
    assert data.shape == expected_shape, f"Data shape mismatch. Expected {expected_shape}, got {data.shape}."
    print("Data shape is correct.")


def test_model(model, input_dim, output_dim, input_length, lstm_units, num_classes):
    """
    Test the BiLSTM model for correct architecture and dimensions.

    Args:
        model: The Keras model instance to test.
        input_dim (int): Size of the vocabulary for the embedding layer.
        output_dim (int): Dimension of the embedding output.
        input_length (int): Number of timesteps in input sequences.
        lstm_units (int): Number of LSTM units in the first BiLSTM layer.
        num_classes (int): Number of output classes (tags).

    Raises:
        AssertionError: If any of the checks fail.
    """
    # Check the input shape
    assert model.input_shape == (None, input_length), \
        f"Expected input shape (None, {input_length}), got {model.input_shape}"
    
    # Check the output shape
    assert model.output_shape == (None, input_length, num_classes), \
        f"Expected output shape (None, {input_length}, {num_classes}), got {model.output_shape}"
    
    # Check the embedding layer
    embedding_layer = model.layers[1]
    assert embedding_layer.input_dim == input_dim, \
        f"Embedding layer input_dim mismatch. Expected {input_dim}, got {embedding_layer.input_dim}"
    assert embedding_layer.output_dim == output_dim, \
        f"Embedding layer output_dim mismatch. Expected {output_dim}, got {embedding_layer.output_dim}"
    
    # Check the first BiLSTM layer
    bidirectional_layer = model.layers[3]
    lstm_layer = bidirectional_layer.forward_layer  # Access the forward LSTM layer within Bidirectional
    assert lstm_layer.units == lstm_units, \
        f"LSTM units mismatch. Expected {lstm_units}, got {lstm_layer.units}"
    
    # Check the number of trainable parameters
    total_params = model.count_params()
    assert total_params > 0, "Model has no trainable parameters!"
    
    print(f"Model test passed with {total_params} trainable parameters!")

def test_validation_accuracy(history, min_val_acc=0.25):
    """
    Checks whether the validation accuracy is at least the specified threshold.
    If not, raises a ValueError.

    Args:
        val_acc (float): The validation accuracy to be checked.
        min_val_acc (float, optional): The minimum acceptable validation accuracy. Defaults to 0.25.

    Raises:
        ValueError: If the validation accuracy is below the threshold.
    """
    val_acc = max(history.history['val_accuracy'])
    if val_acc < min_val_acc:
        raise ValueError(f"Validation accuracy is too low: {val_acc:.2f} (< {min_val_acc}). "
                         f"Please improve the model performance.")
    else:
        print(f"Validation accuracy is sufficient: {val_acc:.2f} (>= {min_val_acc}).")
