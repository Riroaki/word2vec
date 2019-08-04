import jieba
from hierachical.word2vec import CBOW as HSCBOW
from hierachical.word2vec import SKIPGRAM as HSSKIPGRAM
from negative.word2vec import CBOW as NEGCBOW
from negative.word2vec import SKIPGRAM as NEGSKIPGRAM

# Load data from file
with open('test.txt', 'r') as f:
    words = list(jieba.cut(f.read()))

# Form model
# Other choices: HSSKIPGRAM, NEGCBOW, NEGSKIPGRAM
model = HSCBOW(words, vector_dim=50)

# Window size: context size = 4 * 2 = 8 words
window_size = 4

# Load train data
for i in range(window_size, len(words) - window_size):
    context = words[i - window_size: i + window_size]
    target_word = context.pop(len(context) // 2)
    model.backward(target_word, context)

# Save vectors
model.save('50d.csv')
