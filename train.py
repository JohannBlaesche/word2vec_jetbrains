from word2vec.dataset import load_corpus, build_vocab, make_skipgram_pairs
from word2vec.negatives import build_unigram_distribution, sample_negatives
from word2vec.model import SkipGram
import numpy as np


def main():

    tokens = load_corpus("data/sample.txt")

    encoded, word_to_idx, idx_to_word, freqs = build_vocab(tokens)

    pairs = make_skipgram_pairs(encoded, window=2)

    dist = build_unigram_distribution(freqs)

    model = SkipGram(len(word_to_idx), dim=16, lr=0.05)

    rng = np.random.default_rng(0)

    epochs = 50
    neg_k = 4

    print("training examples:", len(pairs))
    print("vocab size:", len(word_to_idx))

    for epoch in range(epochs):
        rng.shuffle(pairs)

        total_loss = 0.0

        for center, pos in pairs:
            neg = sample_negatives(rng, dist, k=neg_k, forbidden={center, pos})

            loss = model.train_example(center, pos, neg)

            total_loss += loss

        avg_loss = total_loss / len(pairs)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print("epoch", epoch + 1, "loss", avg_loss)


if __name__ == "__main__":
    main()
