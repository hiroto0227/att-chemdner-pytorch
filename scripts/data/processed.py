# datsetからcsvを作成する。

import os, sys, re, csv
from tqdm import tqdm
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(CURRENT_DIR, '../'))
from labels import B, I, O, E, S, COMMA, NEWLINE

RAW_DATA_DIR = os.path.join(CURRENT_DIR, '../../datas/raw/')
OUT_DIR = os.path.join(CURRENT_DIR, '../../datas/processed/')

def main(mode='train'):
    # data convert
    fileids = [filename.replace('.txt', '') 
        for filename in  os.listdir(os.path.join(RAW_DATA_DIR, mode)) if filename.endswith('.txt')]
    for fileid in tqdm(fileids):
        with open(os.path.join(RAW_DATA_DIR, mode, fileid + '.txt'), 'rt') as f:
            text = f.read()
        with open(os.path.join(RAW_DATA_DIR, mode, fileid + '.ann'), 'rt') as f:
            annotations = f.read().split('\n')
        tokens, labels = text_ann_to_labels(text, annotations)
        translate_table = str.maketrans({',': COMMA, '\n': NEWLINE})
        tokens = [token.translate(translate_table) for token in tokens]
        if os.path.exists(os.path.join(OUT_DIR, mode + '.csv')):
            with open(os.path.join(OUT_DIR, mode + '.csv'), 'at') as f:
                f.write('\n{},{}'.format(','.join(tokens), ','.join(labels)))
        else:
            with open(os.path.join(OUT_DIR, mode + '.csv'), 'wt') as f:
                f.write('{},{}'.format(','.join(tokens), ','.join(labels)))


def text_ann_to_labels(text, annotations):
    tokens, labels = [], []
    spantokens = text_to_spantokens(text)
    ann_spantokens = annotations_to_spantokens(annotations)
    ann_idx = 0
    for spantoken in spantokens:
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


def to_annfiles(csv_file, out_dir):
    df = pd.read_csv(csv_file)
    for fileid in df.fileids:
        spantokens = labels_to_spantokens(df[df.fileid == fileid].label)
        os.mkdir(out_dir)
        with open(os.path.join(out_dir, fileid + '.ann'), 'wt') as f:
            f.write('\t'.join(spantokens))

def tokenize(text):
    """textをtoken単位に分割したリストを返す。"""
    tokens = re.split("""( | |\xa0|\t|\n|…|\'|\"|·|~|↔|•|\!|@|#|\$|%|\^|&|\*|
        -|=|_|\+|ˉ|\(|\)|\[|\]|\{|\}|;|‘|:|“|,|\.|\/|<|>|×|>|<|≤|≥|↑|↓|¬
        |®|•|′|°|~|≈|\?|Δ|÷|≠|‘|’|“|”|§|£|€|0|1|2|3|4|5|6|7|8|9|™|⋅|-)""", text)
    return list(filter(None, tokens))

def text_to_spantokens(text):
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

def text_to_labels(text, spantokens):
    text_spantokens = text_to_spantokens(text)
    labels = [O for i in range(len(text_spantokens))]
    ann_ix = 0
    ann_spans = sorted([(start, end) for _, start, end in spantokens], key=lambda x: x[0])
    for i, (entity, start, end) in enumerate(text_spantokens):
        if ann_ix == len(ann_spans):
            break
        # startがann_ixがさすendより過ぎた時にはann_ixをincrementする。
        if ann_spans[ann_ix][1] < start:
            ann_ix += 1
        elif start == ann_spans[ann_ix][0] and end == ann_spans[ann_ix][1]:
            labels[i] = S
            ann_ix += 1
        elif start == ann_spans[ann_ix][0] and end < ann_spans[ann_ix][1]:
            labels[i] = B
        elif end == ann_spans[ann_ix][1] and start > ann_spans[ann_ix][0]:
            labels[i] = E
            ann_ix += 1
        elif start > ann_spans[ann_ix][0] and end < ann_spans[ann_ix][1]:
            labels[i] = I
        else:
            pass
    return labels

def labels_to_spantokens(labels, text_spantokens):
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

if __name__ == '__main__':
    for mode in ['train', 'test']:
        main(mode=mode)