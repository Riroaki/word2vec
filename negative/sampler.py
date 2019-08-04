from collections import Counter
import numpy as np


class NegativeSampler(object):
    """Negative Sampling model."""

    TABLE_SIZE = int(1e8)

    def __init__(self, words: list):
        total = len(words)
        word_weight = {word: np.power(count / total, 0.75) for word, count in
                       Counter(words).items()}
        total_weight = sum(word_weight.values())
        table = []
        for word, weight in word_weight.items():
            current = weight / total_weight
            for _ in range(int(NegativeSampler.TABLE_SIZE * current)):
                table.append(word)
        while len(table) < NegativeSampler.TABLE_SIZE:
            random_word = words[np.random.randint(0, len(words))]
            table.append(random_word)
        self._table = table

    def sample(self, target_word: str, neg_count: int) -> list:
        samples = {target_word}
        while len(samples) < neg_count:
            index = np.random.randint(0, NegativeSampler.TABLE_SIZE - 1)
            word = self._table[index]
            samples.add(word)
        return list(samples)
