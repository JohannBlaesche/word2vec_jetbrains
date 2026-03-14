import numpy as np


def sigmoid(x):
    x = np.clip(x, -15, 15)
    return 1 / (1 + np.exp(-x))


class SkipGram:
    def __init__(self, vocab_size, dim=20, lr=0.05):

        self.vocab_size = vocab_size
        self.dim = dim
        self.lr = lr

        # embeddings for center words
        self.W_in = np.random.randn(vocab_size, dim) * 0.01

        # embeddings for context words
        self.W_out = np.random.randn(vocab_size, dim) * 0.01

    def train_example(self, center, pos, negatives):
        """
        Performs one training step for a single skip-gram example.
        """

        v = self.W_in[center]
        u_pos = self.W_out[pos]
        u_neg = self.W_out[negatives]

        # ---- forward ----

        pos_score = np.dot(u_pos, v)
        neg_scores = np.dot(u_neg, v)

        pos_sig = sigmoid(pos_score)
        neg_sig = sigmoid(-neg_scores)

        loss = -np.log(pos_sig + 1e-10) - np.sum(np.log(neg_sig + 1e-10))

        # ---- gradients ----

        g_pos = pos_sig - 1
        g_neg = sigmoid(neg_scores)

        grad_u_pos = g_pos * v
        grad_u_neg = g_neg[:, None] * v

        grad_v = g_pos * u_pos + np.sum(g_neg[:, None] * u_neg, axis=0)

        # ---- update ----

        self.W_in[center] -= self.lr * grad_v
        self.W_out[pos] -= self.lr * grad_u_pos
        self.W_out[negatives] -= self.lr * grad_u_neg

        return loss

    def get_embeddings(self):
        return self.W_in
