from queue import PriorityQueue
import numpy as np


class HuffmanNode(object):
    """Node of a Huffman tree in word2vec (hierachical softmax)."""

    def __init__(self, dim: int, word: str = None, left: 'HuffmanNode' = None,
                 right: 'HuffmanNode' = None):
        self.word = word
        self.vec = np.random.random(dim)
        self.is_leaf = word is not None
        self.left_child = left
        self.right_child = right
        self.encode = ''


class HuffmanTree(object):
    """Huffman Tree in word2vec (hierachical softmax)."""

    LEFT_ENCODE = '0'
    RIGHT_ENCODE = '1'

    def __init__(self, frequency_dict: dict, vector_dim: int):
        heap = PriorityQueue(maxsize=len(frequency_dict))
        index = 0  # Used to compare when frequencies are same
        for word, frequency in frequency_dict.items():
            node = HuffmanNode(dim=vector_dim, word=word)
            heap.put((frequency, index, node))
            index += 1
        while heap.qsize() > 1:
            freq1, _, node1 = heap.get()
            freq2, _, node2 = heap.get()
            parent = HuffmanNode(dim=vector_dim, left=node1, right=node2)
            heap.put((freq1 + freq2, index, parent))
            index += 1
        self.root = heap.get()[-1]
        self.words_dict = {}
        self._encode(self.root, '')

    def _encode(self, root: HuffmanNode, root_encode: str):
        root.encode = root_encode
        if not root.is_leaf:
            self._encode(root.left_child,
                         root_encode + HuffmanTree.LEFT_ENCODE)
            self._encode(root.right_child,
                         root_encode + HuffmanTree.RIGHT_ENCODE)
        else:
            self.words_dict[root.word] = root
