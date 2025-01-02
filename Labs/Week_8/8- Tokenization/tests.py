from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.decoders import ByteLevel

def test_load_data(dataset):
    try:
        assert dataset is not None, "Dataset should not be None."
        assert isinstance(dataset, list), "Dataset should be a list of text entries."
        assert len(dataset) > 0, "Dataset should not be empty."
        print("Test passed: Data loaded successfully!")
    except Exception as e:
        raise ValueError(f"Exception raised: {str(e)}")

def test_initialize_bpe_tokenizer(tokenizer):
    assert isinstance(tokenizer.model, models.BPE), "Tokenizer model should be of type BPE."
    assert isinstance(tokenizer.pre_tokenizer, pre_tokenizers.Whitespace), \
        "Tokenizer pre-tokenizer should be Whitespace."
    print("Test passed: Tokenizer is initialized with BPE model and Whitespace pre-tokenizer.")

def test_train_bpe_tokenizer(tokenizer):
    """
    Test if the tokenizer contains the expected special tokens in its vocabulary.
    """
    # Define the expected special tokens
    expected_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    # Get the tokenizer's vocabulary
    vocab = tokenizer.get_vocab()

    # Check if all expected special tokens are in the vocabulary
    missing_tokens = [token for token in expected_special_tokens if token not in vocab]
    assert not missing_tokens, f"Missing special tokens in tokenizer: {missing_tokens}"

    print("Test passed: Tokenizer contains all expected special tokens.")

def test_tokenizer_func(tokens, ids):
    expected_tokens = ['[CLS]', 'Natural', 'Language', 'Pro', 'cess', 'ing', 'is', 'fascinating', '.', '[SEP]']
    expected_ids = [2, 9634, 19539, 2101, 1379, 1035, 1034, 23616, 18, 3]
    assert tokens == expected_tokens, "Tokens generated and expected tokens do not match"
    assert ids == expected_ids, "IDs generated and expected IDs do not match"

    




