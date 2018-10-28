import sentencepiece as spm


class SentencePieceTokenizer():
    def __init__(self):
        self.spe = spm.SentencePieceProcessor()

    def train(self, text_path, vocab_size):
        spm.SentencePieceTrainer.Train('--input={} --model_prefix={} --vocab_size={} --model_type=unigram'.format(text_path, vocab_size, vocab_size))

    def load(self, sp_model_path):
        self.spe.Load(sp_model_path)

    def tokenize(self, text):
        return [token.replace('▁', '') for token in self.spe.EncodeAsPieces(text)]
