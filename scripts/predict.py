import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert text into annotation files")
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--out_dir', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--fields_path', type=str)
    opt = parser.parse_args()

    train_dataset = ChemdnerSubwordDataset(path=os.path.join(CURRENT_DIR, '../datas/raw/train'), subword_tokenizers=subword_tokenizers)

    for fileid in fileids:
        text = read(fileid)
        pred_labels = predict(text)
        write_annotation_file(pred_labels, os.path.join(opt.out_dir, "{}_pred.ann".format(fileid)))

