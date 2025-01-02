
def load_datasets_test(train_data, test_data):
    assert len(train_data.data) > 0, "Training dataset is empty."
    assert len(test_data.data) > 0, "Test dataset is empty."


def preprocess_text_test(preprocessed_first_example):
    # Define the expected output
    expected = 'wondering anyone could enlighten car saw day sport car looked late early called bricklin door really small addition front bumper separate rest body know anyone tellme model name engine spec year production car made history whatever info funky looking car please'
    
    # Assert that the preprocessed example matches the expected output
    assert preprocessed_first_example == expected, f"Test failed! Expected: {expected}, but got: {preprocessed_first_example}"

def check_vectorizer_test(vectorizer):
    # Check if max_features is 10000
    assert vectorizer.max_features == 10000, f"Test failed! Expected max_features=10000, but got {vectorizer.max_features}"
    
    # Check if ngram_range is (1, 2)
    assert vectorizer.ngram_range == (1, 2), f"Test failed! Expected ngram_range=(1, 2), but got {vectorizer.ngram_range}"
    
    # If both checks pass
    print("Vectorizer settings are correct!")

def check_train_lda_test(lda):
    # Check if learning_decay is 0.7
    assert lda.learning_decay == 0.7, f"Test failed! Expected learning_decay=0.7, but got {lda.learning_decay}"
    
    # Check if max_iter is 10
    assert lda.max_iter == 10, f"Test failed! Expected max_iter=10, but got {lda.max_iter}"
    
    # Check if random_state is 42
    assert lda.random_state == 42, f"Test failed! Expected random_state=42, but got {lda.random_state}"
    
    # Check if n_jobs is -1
    assert lda.n_jobs == -1, f"Test failed! Expected n_jobs=-1, but got {lda.n_jobs}"
    
    # If all checks pass
    print("LDA model parameters are correct!")

