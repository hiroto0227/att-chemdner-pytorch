import os
import torch
import torchtext
from torchtext.data import Field
from tqdm import tqdm
from labels import PAD, UNK, COMMA, NEWLINE
from datautils import char2token, file2sequences, file2char_level_sequences


class TokenizeDataset(torchtext.data.Dataset):
    def __init__(self, path, tokenizer, **kwargs):
        self.token_field = Field(sequential=True, tensor_type=torch.LongTensor, use_vocab=True)
        self.label_field = Field(sequential=True, use_vocab=True)
        examples = []
        for i, fileid in tqdm(enumerate([filename.replace('.txt', '') for filename in os.listdir(path) if filename.endswith('.txt')])):
            #if i == 200:
            #    break
            token_sequence, label_sequence = file2sequences(path, fileid, tokenizer)
            examples.append(torchtext.data.Example.fromlist([token_sequence, label_sequence],
                                                            [('token', self.token_field), ('label', self.label_field)]))
        super(TokenizeDataset, self).__init__(examples, [('token', self.token_field), ('label', self.label_field)], **kwargs)


class ChemdnerSubwordDataset(torchtext.data.Dataset):
    def __init__(self, path, subword_tokenizers={}, fields=None, tokenized_padding="copy", **kwargs):
        """LabelはCharacter Baseで分割し、tokenizerによって分かれたtokenを積み上げていく。
        input:
            subword_tokenizers: 複数のtokenizerを使用する。
            tokenized_padding: "copy" or "zero"
                tokenizeでcharをまとめ上げた時に、最後のtoken以外のtokenを複製するか否か。
                copyの場合の例:
                    "A" "u" "t"  "o" "m"  "a"  "t"  "i"  "c"
                    -> "Auto" "Auto" "Auto" "Auto" "ma" "ma" "tic" "tic" "tic"
        """
        if fields:
            self.fields = fields
        else:
            self.fields = [('char', Field(sequential=True, tensor_type=torch.LongTensor, use_vocab=True))]
            for name in subword_tokenizers.keys():
                self.fields.append((name, Field(sequential=True, tensor_type=torch.LongTensor, use_vocab=True)))
            self.fields.append(('label', Field(sequential=True)))
        examples = []
        for i, fileid in tqdm(enumerate([filename.replace('.txt', '') for filename in os.listdir(path) if filename.endswith('.txt')])):
            if i == 10:
                break
            char_sequence, label_sequence = file2char_level_sequences(path, fileid)
            subword_sequences = [char2token(char_sequence, tokenizer, tokenized_padding="copy") for tokenizer in subword_tokenizers.values()]
            examples.append(torchtext.data.Example.fromlist([char_sequence] + subword_sequences + [label_sequence], self.fields))
        super(ChemdnerSubwordDataset, self).__init__(examples, self.fields, **kwargs)

    def make_vocab(self, load_vector_field_names=[]):
        for field_name, field in self.fields.items():
            field.build_vocab(self)
            if field_name in load_vector_field_names:
                self.text_field.vocab.load_vectors("fasttext.simple.300d")


class ChemdnerDataset(torchtext.data.Dataset):
    """Chemdnerのデータセット
    how to use:
        train_dataset = ChemdnerDataset('./datas/processed/train.csv')
        train_iter = data.Iterator(dataset=train, batch_size=32, shuffle=False, repeat=False)
    """

    def __init__(self, path, fields=None, **kwargs):
        if not fields:
            self.text_field = Field(sequential=True, tensor_type=torch.LongTensor, use_vocab=True)
            self.label_field = Field(sequential=True)
            fields = [('text', self.text_field), ('label', self.label_field)]

        examples = []
        with open(path) as f:
            rows = f.read().split('\n')
            for i, row in enumerate(rows):
                #if i == 500:
                #    break
                splitted_row = row.split(',')
                length = len(splitted_row) // 2
                tokens, labels = splitted_row[:length], splitted_row[length:]
                examples.append(torchtext.data.Example.fromlist([tokens, labels], fields))
        super(ChemdnerDataset, self).__init__(examples, fields, **kwargs)

    def make_vocab(self):
        self.text_field.build_vocab(self)
        self.label_field.build_vocab(self)
        self.text_field.vocab.load_vectors("fasttext.simple.300d")
        return self.text_field.vocab.stoi, self.label_field.vocab.stoi

    def get_id2token(self, tokenize=list):
        id2token = [PAD, UNK, COMMA, '<pad>', NEWLINE]
        for example in self.examples:
            for token in list(''.join(example.text)):
                if not token in id2token:
                    id2token.append(token)
        return id2token
