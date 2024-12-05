
import numpy as np
from nltk.tokenize import word_tokenize
def tokenize_and_preprocess_text(sentence):
    tokens = word_tokenize(sentence)
    tokens=[lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token.lower() for token in tokens if token.isalnum()]

    return tokens

def get_sentence_vector(tokens, model):
    # Filter out tokens that are not in the model's vocabulary
    if not tokens:
        return np.zeros(model.vector_size)
    return np.mean(model.wv[tokens], axis=0)


def embedding_for_vocab(filepath, word_index,
						embedding_dim):
	vocab_size = len(word_index) + 1

	# Adding again 1 because of reserved 0 index
	embedding_matrix_vocab = np.zeros((vocab_size,
									embedding_dim))

	with open(filepath, encoding="utf8") as f:
		for line in f:
			word, *vector = line.split()
			if word in word_index:
				idx = word_index[word]
				embedding_matrix_vocab[idx] = np.array(
					vector, dtype=np.float32)[:embedding_dim]

	return embedding_matrix_vocab

