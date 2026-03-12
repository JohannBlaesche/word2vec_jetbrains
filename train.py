from word2vec.dataset import load_corpus, build_vocab, make_skipgram_pairs


def main():

    tokens = load_corpus("data/sample.txt")

    encoded, word_to_idx, idx_to_word = build_vocab(tokens)

    pairs = make_skipgram_pairs(encoded, window=2)

    print("tokens:", len(tokens))
    print("vocab size:", len(word_to_idx))
    print("training pairs:", len(pairs))

    print("\nfirst few pairs:")
    for p in pairs[:10]:
        print(p)


if __name__ == "__main__":
    main()
