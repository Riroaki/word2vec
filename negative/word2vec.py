import numpy as np
from .sampler import NegativeSampler


def sigmoid(x: float) -> float:
    return 1.0 / (1 + np.exp(-x))


class CBOW(object):
    """Continuous Bag-Of-Words model.
    Based on negative sampling.
    """

    def __init__(self, words: list, vector_dim: int, neg_count: int = 20):
        unique_words = list(set(words))
        self._word_vec = {w: np.random.random(vector_dim) for w in unique_words}
        self._help_vec = {w: np.random.random(vector_dim) for w in unique_words}
        self._sampler = NegativeSampler(words)
        self._dim = vector_dim
        self._neg = neg_count

    def save(self, fname: str) -> None:
        with open(fname, 'w') as f:
            f.write('word\tvector\n')
            for word, vector in self._word_vec.items():
                f.write('{}\t{}\n'.format(word, vector))

    def forward(self, target_word: str, context_words: list) -> float:
        context_vec = np.zeros(self._dim)
        for word in context_words:
            context_vec += self._word_vec[word]
        probability = 1.
        for word in self._sampler.sample(target_word, self._neg):
            vector = self._help_vec[word]
            if word == target_word:
                prob = 1 - sigmoid(np.dot(context_vec, vector))
            else:
                prob = sigmoid(np.dot(context_vec, vector))
            probability *= prob
        return probability

    def backward(self, target_word: str, context_words: list,
                 lr: float = 2e-3) -> None:
        context_vec = np.zeros(self._dim)
        for word in context_words:
            context_vec += self._word_vec[word]
        for word in self._sampler.sample(target_word, self._neg):
            vector = self._help_vec[word]
            is_positive_sample = int(word == target_word)
            prob = sigmoid(np.dot(context_vec, vector))
            for context_word in context_words:
                self._word_vec[context_word] += lr * (
                        is_positive_sample - prob) * vector
            self._help_vec[word] += lr * (
                    is_positive_sample - prob) * context_vec


class SKIPGRAM(object):
    """Skip Gram model.
    Based on negative sampling.
    """

    def __init__(self, words: list, vector_dim: int, neg_count: int):
        unique_words = list(set(words))
        self._word_vec = {w: np.random.random(vector_dim) for w in unique_words}
        self._help_vec = {w: np.random.random(vector_dim) for w in unique_words}
        self._sampler = NegativeSampler(words)
        self._neg = neg_count
        self._dim = vector_dim

    def save(self, fname: str) -> None:
        with open(fname, 'w') as f:
            f.write('word\tvector\n')
            for word, vector in self._word_vec.items():
                f.write('{}\t{}\n'.format(word, vector))

    def forward(self, target_word: str, context_words: list) -> float:
        probability = 1.
        for context_word in context_words:
            context_vec = self._word_vec[context_word]
            for word in self._sampler.sample(target_word, self._neg):
                vector = self._help_vec[word]
                if word == target_word:
                    prob = sigmoid(np.dot(vector, context_vec))
                else:
                    prob = 1 - sigmoid(np.dot(vector, context_vec))
                probability *= prob
        return probability

    def backward(self, target_word: str, context_words: list,
                 lr: float = 2e-3) -> None:
        for context_word in context_words:
            context_vec = self._word_vec[context_word]
            for word in self._sampler.sample(target_word, self._neg):
                vector = self._help_vec[word]
                is_positive_sample = int(word == target_word)
                prob = sigmoid(np.dot(vector, context_vec))
                self._word_vec[context_word] += lr * (
                        is_positive_sample - prob) * vector
                self._help_vec[word] += lr * (
                        is_positive_sample - prob) * context_vec
