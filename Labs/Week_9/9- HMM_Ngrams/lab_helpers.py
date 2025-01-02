
def autocomplete(text, bigram_model, num_words=5):
    words = text.split()
    last_word = words[-1].lower()
    predictions = []

    for _ in range(num_words):
        if last_word in bigram_model:
            next_word = bigram_model[last_word].most_common(1)[0][0]
            if next_word == "</s>":
                break
            predictions.append(next_word)
            last_word = next_word
        else:
            break

    return text + " " + " ".join(predictions)

def autocomplete_HMM(text, bigram_model, num_words=5):
    words = text.split()
    last_word = words[-1].lower()
    predictions = []

    for _ in range(num_words):
        if last_word in bigram_model:
            next_word = bigram_model[last_word].most_common(1)[0][0]
            if next_word == "</s>":
                break
            predictions.append(next_word)
            last_word = next_word
        else:
            break

    return text + " " + " ".join(predictions)


import nltk
from nltk.corpus import treebank, words
from nltk.tag import hmm
from nltk.metrics.distance import edit_distance

def suggest_corrections(word, candidates, max_distance=2):
    suggestions = [candidate for candidate in candidates if edit_distance(word, candidate) <= max_distance]
    if not suggestions:
        suggestions = [candidate for candidate in candidates if edit_distance(word, candidate) <= max_distance + 1]
    return min(suggestions, key=lambda candidate: edit_distance(word, candidate), default=word)


