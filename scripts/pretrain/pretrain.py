import numpy as np


def load_word2vec(word2vec_path):
    with open(word2vec_path, "r") as f:
        word2vec = {}
        for line in f:
            if line:
                line_splited = line.split()
                word, vec = line_splited[0], line_splited[1:]
                try:
                    word2vec[word] = np.array(vec, dtype=np.float32)
                except ValueError:
                    print("ValueError: {}".format(word))
    return word2vec


def make_pretrain_embed(word2vec, token2id, word_embed_size):
    embed = np.zeros((len(token2id), word_embed_size))
    for token, idx in token2id.items():
        try:
            embed[idx] = word2vec[token]
        except KeyError:
            try:
                embed[idx] = word2vec[token.lower()]
            except KeyError:
                print("OOV: {}".format(token))
        except ValueError:
            print("=" * 50)
            print(token)
            print(word2vec[token].shape)
    return embed
