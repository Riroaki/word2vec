from collections import Counter
import numpy as np
from .tree import HuffmanTree


def sigmoid(x: float) -> float:
    return 1.0 / (1 + np.exp(-x))


class CBOW(object):
    """Continuous Bag-Of-Words model.
    Based on hierachical softmax and Huffman tree.
    """

    def __init__(self, words: list, vector_dim: int):
        frequency_dict = Counter(words)
        self._dim = vector_dim
        self._tree = HuffmanTree(frequency_dict, vector_dim)

    def save(self, fname: str):
        with open(fname, 'w') as f:
            f.write('word\tvector\n')
            for word, node in self._tree.words_dict:
                vector = node.vec
                f.write('{}\t{}\n'.format(word, vector))

    def forward(self, target_word: str, context_words: list) -> float:
        context_vec = np.zeros(self._dim)
        for word in context_words:
            context_vec += self._tree.words_dict[word].vec
        encode = self._tree.words_dict[target_word].encode
        probability = 1.
        node = self._tree.root
        for c in encode:
            d = 1 - int(c == self._tree.LEFT_ENCODE)
            prob = sigmoid(np.dot(node.vec, context_vec))
            probability *= (1 - d) * prob + d * (1 - prob)
            if d == 0:
                node = node.left_child
            else:
                node = node.right_child
        return probability

    def backward(self, target_word: str, bag_words: list,
                 lr: float = 2e-3) -> None:
        context_vec = np.zeros(self._dim)
        for word in bag_words:
            context_vec += self._tree.words_dict[word].vec
        encode = self._tree.words_dict[target_word].encode
        node = self._tree.root
        for c in encode:
            prob = sigmoid(np.dot(node.vec, context_vec))
            d = 1 - int(c == self._tree.LEFT_ENCODE)
            node.vec += lr * context_vec * (1 - d - prob)
            word_grad = lr * node.vec * (1 - d - prob)
            for word in bag_words:
                self._tree.words_dict[word].vec += word_grad
            if d == 0:
                node = node.left_child
            else:
                node = node.right_child


class SKIPGRAM(object):
    """Skip Gram model.
    Based on hierachical softmax and Huffman tree.
    """

    def __init__(self, words: list, vector_dim: int):
        frequency_dict = Counter(words)
        self._dim = vector_dim
        self._tree = HuffmanTree(frequency_dict, vector_dim)

    def save(self, fname: str = None):
        with open(fname, 'w') as f:
            f.write('word\tvector\n')
            for word, node in self._tree.words_dict:
                vector = node.vec
                line = '{}\t{}\n'.format(word, vector)
                f.write(line)

    def forward(self, target_word: str, context_words: list) -> float:
        probability = 1.
        target_vec = self._tree.words_dict[target_word].vec
        for word in context_words:
            prob_context = 1.
            node = self._tree.root
            encode = self._tree.words_dict[word].encode
            for c in encode:
                d = 1 - int(c == self._tree.LEFT_ENCODE)
                prob = sigmoid(np.dot(node.vec, target_vec))
                prob_context *= (1 - d) * prob + d * (1 - prob)
                if d == 0:
                    node = node.left_child
                else:
                    node = node.right_child
            probability *= prob_context
        return probability

    def backward(self, target_word: str, context_words: list,
                 lr: float = 2e-3) -> None:
        target_vec = self._tree.words_dict[target_word].vec
        for word in context_words:
            node = self._tree.root
            encode = self._tree.words_dict[word].encode
            for c in encode:
                prob = sigmoid(np.dot(node.vec, target_vec))
                d = 1 - int(c == self._tree.LEFT_ENCODE)
                node.vec += lr * target_vec * (1 - d - prob)
                target_grad = lr * node.vec * (1 - d - prob)
                self._tree.words_dict[word].vec += target_grad
                if d == 0:
                    node = node.left_child
                else:
                    node = node.right_child
