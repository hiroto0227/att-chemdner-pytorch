import os
import re
from seqeval.metrics.sequence_labeling import get_entities
from labels import O, S, B, I, E

ok = 0
fail = 0


def file2sequences(path, fileid, tokenizer):
    global fail, ok
    with open(os.path.join(path, fileid + '.txt'), 'rt') as f:
        text = f.read()
    with open(os.path.join(path, fileid + '.ann'), 'rt') as f:
        annotations = f.read().split('\n')
    text_spantokens = text_to_spantokens(text, tokenizer)
    ann_spantokens = annotations_to_spantokens(annotations)
    token_sequence, label_sequence = make_tokens_and_labels(text_spantokens, ann_spantokens)
    ############# test ##############
    pred_spantokens = labels_to_anns(label_sequence, text_spantokens)
    ok += len(ann_spantokens)
    if ann_spantokens != pred_spantokens:
        #print('#########################')
        #print(text_spantokens)
        #print('--------------------------')
        #print(label_sequence)
        #print('--------------------------')
        #print(ann_spantokens)
        #print('--------------------------')
        #print(pred_spantokens)
        fail += len(set(ann_spantokens) - set(pred_spantokens))
        ok -= len(set(ann_spantokens) - set(pred_spantokens))
        pass
        #print(ok)
        print("labelize fail num: {}".format(fail))
    else:
        #print('================================')
        #print(ann_spantokens)
        #print('----------------------------------------')
        #print(pred_spantokens)
        pass
    #################################
    return token_sequence, label_sequence


def make_tokens_and_labels(text_spantokens, ann_spantokens):
    tokens, labels = [], []
    ann_idx = 0
    for spantoken in text_spantokens:
        if ann_idx >= len(ann_spantokens):
            tokens.append(spantoken[0])
            labels.append(O)
        elif (spantoken[1] == ann_spantokens[ann_idx][1] and spantoken[2] == ann_spantokens[ann_idx][2]):
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
            if labels[-1] == O:
                tokens.append(spantoken[0])
                labels.append(B)
            else:
                tokens.append(spantoken[0])
                labels.append(I)
        else:
            # tokenizerによりラベルがうまく分割されなかった時。
            if spantoken[1] <= ann_spantokens[ann_idx][1] and spantoken[2] >= ann_spantokens[ann_idx][2]:
                ann_idx += 1
                tokens.append(spantoken[0])
                labels.append(S)
            elif len(labels) > 0 and labels[-1] in [B, I]:
                ann_idx += 1
                tokens.append(spantoken[0])
                labels.append(E)
            else:
                if spantoken[1] > ann_spantokens[ann_idx][2]:
                    ann_idx += 1
                tokens.append(spantoken[0])
                labels.append(O)
    assert len(tokens) == len(labels), 'tokensとlabelsの値が違います。'
    return tokens, labels


def text_to_spantokens(text, tokenizer):
    """textをtokenizeし、(token, start_ix, end_ix)のリストとして返す。"""
    spantokens = []
    ix = 0
    for token in tokenizer(text):
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
    return sorted(spantokens, key=lambda x: x[1])


def labels_to_anns(labels, text_spantokens):
    spantokens = set()
    entity = ''
    ann_start = 0
    pre_label = O
    for text_ann, label in zip(text_spantokens, labels):
        pre_now = (pre_label, label)
        # あり得るラベル列の組み合わせ
        if pre_now in [(B, I), (B, E), (I, E), (E, B), (E, S), (I, I),
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
    return sorted(spantokens, key=lambda x: x[1])


def char2token(char_sequence, tokenizer, tokenized_padding="copy"):
    token_sequence = []
    if tokenized_padding == "copy":
        for token in tokenizer("".join(char_sequence)):
            token_sequence.extend([token for i in range(len(token))])
        assert len(char_sequence) == len(token_sequence), "There are not equal length. (char_sequence, token_sequence)"
        return token_sequence
    elif tokenized_padding == "none":
        return [token for token in tokenizer("".join(char_sequence))]


def tokenize(text):
    """textをtoken単位に分割したリストを返す。"""
    tokens = re.split("""( | |\xa0|\t|\n|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|-|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\.|\/|<|>|×|>|<|≤|≥|↑|↓|→|¬|®|•|′|°|~|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|0|1|2|3|4|5|6|7|8|9|™|⋅|-|\u2000|⁺|\u2009|)""", text)
    return list(filter(None, tokens))


def file2char_level_sequences(path, fileid):
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
    ######## labelize test(valid で間違えることあり。)) ###############
    #print('======================')
    #print(label_sequence)
    #print('=====================')
    #print(annotation2entities(annotation))
    #print(get_entities(label_sequence))
    #true_labels = [(start, end) for _, start, end in annotation2entities(annotation)]
    #encoded_labels = [(start, end) for _, start, end in get_entities(label_sequence)]
    #assert true_labels == encoded_labels, "Not valid Label Encoding"
    return char_sequence, label_sequence
