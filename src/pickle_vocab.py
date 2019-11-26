#!/usr/bin/env python3
import pickle

data_path = '../data/'


def main():
    vocab = dict()
    with open(data_path + 'vocab_cut.txt') as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx

    with open(data_path + 'vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
