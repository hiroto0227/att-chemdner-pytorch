import os
from labels import B, I, O, E, S


def file2sequences(path, fileid, tokenizer):
    with open(os.path.join(path, fileid + '.txt'), 'rt') as f:
        text = f.read()
    with open(os.path.join(path, fileid + '.ann'), 'rt') as f:
        annotations = f.read().split('\n')
    text_spantokens = text_to_spantokens(text)
    ann_spantokens = annotations_to_spantokens(annotations)
    token_sequence, label_sequence = make_tokens_and_labels(text_spantokens, ann_spantokens)
    ############# test ##############
    pred_spantokens = labels_to_anns(label_sequence)
    for ann_spantoken, pred_spantoken in zip(ann_spantokens, pred_spantokens):
        if ann_spantoken != pred_spantoken:
            print('================================')
            print(ann_spantoken)
            print('###############################')
            print(pred_spantoken)
    #################################
    print("------- end tokenize --------------")
    return token_sequence, label_sequence


def make_tokens_and_labels(text_spantokens, ann_spantokens):
    tokens, labels = [], []
    ann_idx = 0
    for spantoken in text_spantokens:
        if ann_idx >= len(ann_spantokens) - 1:
            tokens.append(spantoken[0])
            labels.append(O)
        elif spantoken[1] == ann_spantokens[ann_idx][1] and spantoken[2] == ann_spantokens[ann_idx][2]:
            ann_idx += 1
            tokens.append(spantoken[0])
            labels.append(S)
        elif spantoken[1] == ann_spantokens[ann_idx][1]:
            tokens.append(spantoken[0])
            labels.append(B)
        elif spantoken[2] == ann_spantokens[ann_idx][2]:
            ann_idx += 1
            tokens.append(spantoken[0])
            labels.append(E)
        elif spantoken[1] > ann_spantokens[ann_idx][1] and spantoken[2] < ann_spantokens[ann_idx][2]:
            tokens.append(spantoken[0])
            labels.append(I)
        else:
            tokens.append(spantoken[0])
            labels.append(O)
    assert len(tokens) == len(labels), 'tokensとlabelsの値が違います。'
    return tokens, labels


def text_to_spantokens(text, tokenize):
    """textをtokenizeし、(token, start_ix, end_ix)のリストとして返す。"""
    spantokens = []
    ix = 0
    for token in tokenize(text):
        spantokens.append((token, ix, ix + len(token)))
        ix += len(token)
    return spantokens


def annotations_to_spantokens(annotations):
    spantokens = []
    for annotation in annotations:
        if annotation:
            token = annotation.split('\t')[-1]
            start = int(annotation.split('\t')[1].split(' ')[1])
            end = int(annotation.split('\t')[1].split(' ')[-1])
            if token:
                spantokens.append((token, start, end))
    return spantokens


def labels_to_anns(labels, text_spantokens):
    spantokens = set()
    entity = ''
    ann_start = 0
    pre_label = ''
    for text_ann, label in zip(text_spantokens, labels):
        pre_now = (pre_label, label)
        # あり得るラベル列の組み合わせ
        if pre_now in [(B, I), (B, E), (I, E), (E, B), (E, S),
                       (S, B), (O, S), (O, B), (O, I), (O, E)]:
            if label == S:
                spantokens.add(text_ann)
            elif label == B:
                ann_start = text_ann[1]
                entity += text_ann[0]
            elif label == I:
                entity += text_ann[0]
            elif label == E:
                entity += text_ann[0]
                spantokens.add((entity, ann_start, text_ann[2]))
                entity = ''
            elif label == O:
                entity += text_ann[0]
        pre_label = label
    return spantokens
