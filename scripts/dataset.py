import torch
import torchtext
from torchtext.data import Field


class ChemdnerDataset(torchtext.data.Dataset):
    """Chemdnerのデータセット
    how to use:
        train_dataset = ChemdnerDataset('./datas/processed/train.csv')
        train_iter = data.Iterator(dataset=train, batch_size=32, shuffle=False, repeat=False)
    """

    def __init__(self, path, fields=None, **kwargs):
        if not fields:
            self.text_field = Field(sequential=True, tensor_type=torch.LongTensor)
            self.label_field = Field(sequential=True)
            fields = [('text', self.text_field), ('label', self.label_field)]

        examples = []
        with open(path) as f:
            rows = f.read().split('\n')
            for i, row in enumerate(rows):
                if i >= 12:
                    break
                splitted_row = row.split(',')
                length = len(splitted_row) // 2
                tokens, labels = splitted_row[:length], splitted_row[length:]
                examples.append(torchtext.data.Example.fromlist([tokens, labels], fields))
        super(ChemdnerDataset, self).__init__(examples, fields, **kwargs)

    def make_vocab(self):
        self.text_field.build_vocab(self)
        self.label_field.build_vocab(self)
        return self.text_field.vocab.stoi, self.label_field.vocab.stoi