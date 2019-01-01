import chemdnerdatautils
from tqdm import tqdm
import dataset
import argparse
from sentencepieces.sp_tokenizer import SentencePieceTokenizer


def to_watanabe_format(token_seqs, tag_seqs, out_path):
   seqs = []
   all_seqs = []
   for token_seq, tag_seq in tqdm(zip(token_seqs, tag_seqs)):
       for token, tag in zip(token_seq, tag_seq):
           #token = token.replace("\n", "")
           if token == "\n":
               continue
           elif token == "." or token == "?" or token == "?\n" or token == "!" or token == "!\n":
               token = token.replace("\n", "")
               seqs.append("{}\t{}".format(token, tag))
               seqs.append("")
               all_seqs.append("\n".join(seqs))
               seqs = []
           else:
               seqs.append("{}\t{}".format(token, tag))
   with open(out_path, "wt") as f:
       print(out_path)
       f.write("\n".join(all_seqs))

def to_watanabe_char_format(token_seqs, tag_seqs, out_path):
   seqs = []
   all_seqs = []
   for token_seq, tag_seq in tqdm(zip(token_seqs, tag_seqs)):
       for token, tag in zip(token_seq, tag_seq):
           #token = token.replace("\n", "")
           if token == "\n":
               continue
           elif token == "." or token == "?" or token == "?\n" or token == "!" or token == "!\n":
               token = token.replace("\n", "")
               seqs.append("{}\t{}".format(token, tag))
               seqs.append("")
               all_seqs.append("\n".join(seqs))
               seqs = []
           else:
               seqs.append("{}\t{}".format(token, tag))
   with open(out_path, "wt") as f:
       print(out_path)
       f.write("\n".join(all_seqs))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--output-path", type=str)
    parser.add_argument('--sp-model', type=str, default=None, help='sentencepiece model path')
    args = parser.parse_args()

    sp = SentencePieceTokenizer()    
    #sp.load(args.sp_model)
    #tokenize = sp.tokenize
    tokenize = chemdnerdatautils.char_tokenize
    #token_seqs, label_seqs = dataset.load_sequences(args.input_path, tokenizeByTweetTokenizer)
    token_seqs, label_seqs = dataset.load_sequences(args.input_path, tokenize)
    to_watanabe_char_format(token_seqs, label_seqs, args.output_path)
