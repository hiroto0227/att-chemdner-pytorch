import os
from seqeval.metrics.sequence_labeling import get_entities
from labels import O, S, B, I, E


def char2token(char_sequence, tokenizer, tokenized_padding="copy"):
    token_sequence = []
    for token in tokenizer("".join(char_sequence)):
        token_sequence.extend([token for i in range(len(token))])
    assert len(char_sequence) == len(token_sequence), "There are not equal length. (char_sequence, token_sequence)"
    return token_sequence


def file2sequences(path, fileid):
    def annotation2entities(annotation):
        entities = []
        for entity in annotation:
            if entity:
                token = entity.split('\t')[-1]
                start = int(entity.split('\t')[1].split(' ')[1])
                # endは.ann上では+1されている
                end = int(entity.split('\t')[1].split(' ')[-1]) - 1
                if token:
                    entities.append((token, start, end))
        return entities

    with open(os.path.join(path, fileid + '.txt'), "r") as f:
        char_sequence = list(f.read())
    with open(os.path.join(path, fileid + '.ann'), "r") as f:
        annotation = f.read().split('\n')
    label_sequence = [O for i in range(len(char_sequence))]
    for entity, start, end in annotation2entities(annotation):
        if start == end:
            label_sequence[start] = S
        else:
            label_sequence[start] = B
            label_sequence[end] = E
            if end - start >= 2:
                for i in range(1, end - start):
                    label_sequence[start + i] = I
    #print('======================')
    #print(label_sequence)
    #print('=====================')
    #print(annotation2entities(annotation))
    #print(get_entities(label_sequence))
    true_labels = [(start, end) for _, start, end in annotation2entities(annotation)]
    encoded_labels = [(start, end) for _, start, end in get_entities(label_sequence)]
    assert true_labels == encoded_labels, "Not valid Label Encoding"
    return char_sequence, label_sequence
