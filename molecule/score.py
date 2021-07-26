import pandas as pd
import io
from rdkit import Chem
import re
import argparse
import os


def detokenize(smi):
    return "".join(smi.split(" "))


def canonicalize_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        return Chem.MolToSmiles(mol,)
    else:
        return ""


def get_rank(row, prefix, max_rank):
    for i in range(1, max_rank + 1):
        preds = row["{}{}".format(prefix, i)].split(".")
        preds.sort()
        labels = row["labels"].split(".")
        labels.sort()
        if preds == labels:
            return i
    return 0


def main(args):
    lines = io.open(args.input_fn, "r", newline="\n", encoding="utf8").readlines()
    lines = [line.strip() for line in lines]

    beam_size = re.findall("'nbest': (\d+),", lines[0])
    beam_size = int(beam_size[0])

    if args.output_fn is None:
        args.output_fn = os.path.splitext(args.input_fn)[0] + ".csv"

    if not args.topk:
        products = []
        labels = []
        classes = []
        hypotheses = [[] for _ in range(beam_size)]
        ptr = 0
        for line in lines:
            if line.startswith("S-"):
                line = line.split("\t")[-1]
                cls, smi = line.split(" ", maxsplit=1)
                smi = detokenize(smi)
                products.append(smi)
                classes.append(cls)
            elif line.startswith("T-"):
                line = line.split("\t")[-1]
                smi = detokenize(line)
                labels.append(smi)
            elif line.startswith("H-"):
                line = line.split("\t")[-1]
                smi = detokenize(line)
                hypotheses[ptr].append(smi)
                ptr = (ptr + 1) % beam_size
        assert ptr == 0
        test_df = pd.DataFrame({"class": classes, "products": products, "labels": labels,})
        total = len(test_df)
        for i in range(1, beam_size + 1):
            test_df["prediction_{}".format(i)] = hypotheses[i - 1]
            test_df["canonical_prediction_{}".format(i)] = test_df["prediction_{}".format(i)].apply(
                canonicalize_smiles
            )
        test_df["rank"] = test_df.apply(
            lambda row: get_rank(row, "canonical_prediction_", beam_size), axis=1
        )
    else:
        test_df = pd.read_csv(args.output_fn)
        total = len(test_df)

    correct = 0
    for i in range(1, beam_size + 1):
        correct += (test_df["rank"] == i).sum()
        invalid_smiles = (test_df["canonical_prediction_{}".format(i)] == "").sum()
        print(
            "Top-{}: {:.1f}% || Invalid SMILES {:.2f}%".format(
                i, correct / total * 100, invalid_smiles / total * 100
            )
        )
    if not args.topk:
        test_df.to_csv(args.output_fn, encoding="utf8", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fn", type=str)
    parser.add_argument("--output-fn", type=str, default=None)
    parser.add_argument("--topk", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
