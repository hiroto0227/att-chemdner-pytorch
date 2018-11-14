import argparse
import sys
sys.path.append("../")
from sentencepieces.sp_tokenizer import SentencePieceTokenizer
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str)
    parser.add_argument("--sp-model", type=str)
    parser.add_argument("--output-path", type=str)
    opt = parser.parse_args()

    sp = SentencePieceTokenizer()
    sp.load(opt.sp_model)

    tokenized_lines = []
    with open(opt.input_path, "rt") as f:
        for i, line in tqdm(enumerate(f.read().split("\n"))):
            tokenized_lines.append(" ".join(sp.tokenize(line)))
    
    with open(opt.output_path, "wt") as f:
        f.write("\n".join(tokenized_lines))
