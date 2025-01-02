import tensorflow as tf
import regex as re

# Preprocess sentences
def preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    sentence = "<start> " + sentence + " <end>"
    return sentence

# Create dataset
def create_dataset(path, num_examples=None):
    with open(path, encoding='utf-8') as f:
        lines = f.read().strip().split('\n')
    sentence_pairs = [[preprocess_sentence(sentence) for sentence in line.split('\t')[:2]] for line in lines]
    return zip(*sentence_pairs[:num_examples])

# Tokenize the data
def tokenize(lang):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(lang)
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer

