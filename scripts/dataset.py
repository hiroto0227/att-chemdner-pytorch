import os
from tqdm import tqdm
import labels
from chemdnerdatautils import file2sequences


def load_sequences(path, tokenizer):
    token_seqs = []
    label_seqs = []
    fileids = [filename.replace('.txt', '') for filename in os.listdir(path) if filename.endswith('.txt')]
    for i, fileid in tqdm(enumerate(fileids)):
        if i == 50:
            break
        token_seq, label_seq = file2sequences(path, fileid, tokenizer)
        token_seqs.append(token_seq)
        label_seqs.append(label_seq)
    return token_seqs, label_seqs


def to_id(seqs, dic, char=False, label=False):
    id_seqs = []
    for seq in seqs:
        if char:
            id_seqs.append([[dic.get(u, dic[labels.UNK]) for u in unit] for unit in seq])
        elif label:
            id_seqs.append([dic.get(unit) for unit in seq])
        else:
            id_seqs.append([dic.get(unit, dic[labels.UNK]) for unit in seq])
    return id_seqs


def batch_gen(token_id_seqs, char_id_seqs, label_id_seqs, batch_size, word_pad_ix, char_pad_ix, label_pad_ix, shuffle=False):
    if shuffle:
        pass
    token_batches, char_batches, label_batches = [], [], []
    max_len = 0
    char_max_len = 0
    for i, (token_id_seq, char_id_seq, label_id_seq) in enumerate(zip(token_id_seqs, char_id_seqs, label_id_seqs)):
        if max_len < len(token_id_seq):
            max_len = len(token_id_seq)
        if char_max_len < max([len(token_ids) for token_ids in char_id_seq]):
            char_max_len = max([len(token_ids) for token_ids in char_id_seq])
        token_batches.append(token_id_seq)
        char_batches.append(char_id_seq)
        label_batches.append(label_id_seq)
        if (i + 1) % batch_size == 0:
            #yield (padding(token_batches, max_len, word_pad_ix),
            #       padding(char_batches, max_len, char_pad_ix, char_level_pad=True),
            #       padding(label_batches, max_len, label_pad_ix))
            yield (padding(token_batches, max_len, word_pad_ix),
                   padding(char_batches, max_len, char_pad_ix, char_level_pad=True,  char_max_len=char_max_len),
                   padding(label_batches, max_len, label_pad_ix))
            token_batches, char_batches, label_batches = [], [], []


def padding(batches, max_len, pad_ix, char_level_pad=False, char_max_len=None):
    pad_batches = []
    for batch in batches:
        pad_length = max_len - len(batch)
        if char_max_len:
            padded_char_batch = []
            for b in batch:
                char_pad_length = char_max_len - len(b)
                padded_char_batch.append(b + [pad_ix for i in range(char_pad_length)])
            pad_batch = [[pad_ix for i in range(char_max_len)] for j in range(pad_length)]
            pad_batches.append(padded_char_batch + pad_batch)
        else:
            if char_level_pad:
                pad_batches.append(batch + [[pad_ix] for i in range(pad_length)])
            else:
                pad_batches.append(batch + [pad_ix for i in range(pad_length)])
    return pad_batches


def make_vocab(token_seqs):
    tokenset = set([token for seq in token_seqs for token in seq])
    token2id = {labels.UNK: 0, labels.PAD: 1}
    token2id.update({token: i + 2 for i, token in enumerate(sorted(tokenset))})
    charset = set([char for chars in tokenset for char in chars])
    char2id = {labels.UNK: 0, labels.PAD: 1}
    char2id.update({token: i + 2 for i, token in enumerate(sorted(charset))})
    print("token vocab dim: {}, char vocab dim: {}".format(len(token2id), len(char2id)))
    return token2id, char2id
