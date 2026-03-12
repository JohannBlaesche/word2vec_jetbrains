import numpy as np
from collections import Counter
from .utils import tokenize


def load_corpus(path):

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenize(text)
    return tokens


def build_vocab(tokens):

    counts = Counter(tokens)

    vocab = sorted(counts.keys())

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    encoded = [word_to_idx[w] for w in tokens]

    return encoded, word_to_idx, idx_to_word


def make_skipgram_pairs(tokens, window=2):

    pairs = []

    n = len(tokens)

    for i in range(n):
        center = tokens[i]

        left = max(0, i - window)
        right = min(n, i + window + 1)

        for j in range(left, right):
            if i == j:
                continue

            context = tokens[j]

            pairs.append((center, context))

    return pairs
