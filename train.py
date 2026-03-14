from word2vec.dataset import load_corpus, build_vocab, make_skipgram_pairs
from word2vec.negatives import build_unigram_distribution, sample_negatives
from word2vec.model import SkipGram
import numpy as np


def main():

    tokens = load_corpus("data/sample.txt")

    encoded, word_to_idx, idx_to_word, freqs = build_vocab(tokens)

    pairs = make_skipgram_pairs(encoded, window=2)

    dist = build_unigram_distribution(freqs)

    model = SkipGram(len(word_to_idx), dim=16)

    rng = np.random.default_rng(0)

    # test one training step
    center, pos = pairs[0]

    neg = sample_negatives(rng, dist, k=4, forbidden={center, pos})

    loss = model.train_example(center, pos, neg)

    print("example loss:", loss)


if __name__ == "__main__":
    main()
