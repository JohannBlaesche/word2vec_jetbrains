from word2vec.dataset import load_corpus, build_vocab, make_skipgram_pairs
from word2vec.negatives import build_unigram_distribution, sample_negatives
import numpy as np


def main():

    tokens = load_corpus("data/sample.txt")

    encoded, word_to_idx, idx_to_word, freqs = build_vocab(tokens)

    pairs = make_skipgram_pairs(encoded, window=2)

    dist = build_unigram_distribution(freqs)

    rng = np.random.default_rng(0)

    print("tokens:", len(tokens))
    print("vocab size:", len(word_to_idx))
    print("training pairs:", len(pairs))

    # example negative sampling
    center, context = pairs[0]

    neg = sample_negatives(rng, dist, k=5, forbidden={center, context})

    print("\nexample negative samples:", neg)


if __name__ == "__main__":
    main()
