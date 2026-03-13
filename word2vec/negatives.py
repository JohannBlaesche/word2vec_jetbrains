import numpy as np


def build_unigram_distribution(freqs):
    """
    Build the negative sampling distribution.

    Word2Vec uses the unigram distribution raised to the power 0.75.
    This makes frequent words appear more often as negatives,
    but not as extremely often as with the raw counts.
    """

    freqs = np.asarray(freqs, dtype=float)

    dist = freqs**0.75
    dist = dist / dist.sum()

    return dist


def sample_negatives(rng, dist, k, forbidden=None):
    """
    Draw k negative samples from the distribution.

    forbidden: optional set of word indices that should not appear
    (for example the center or positive context word)
    """

    vocab_size = len(dist)

    if forbidden is None:
        forbidden = set()

    negatives = []

    while len(negatives) < k:
        idx = rng.choice(vocab_size, p=dist)

        if idx in forbidden:
            continue

        negatives.append(idx)

    return np.array(negatives)
