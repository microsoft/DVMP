import os
import argparse
import io
from fairseq.models.onemodel import OneModel
from sklearn.metrics import (
    roc_curve,
    auc,
    average_precision_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
)
import torch
import torch.nn.functional as F
import math
from math import floor
from collections import Counter


def ret_rmse(targets, preds):
    return math.sqrt(mean_squared_error(targets, preds))


def main(dataset, cktpath, subset):
    roberta = OneModel.from_pretrained(
        os.path.dirname(cktpath),
        checkpoint_file=os.path.basename(cktpath),
        data_name_or_path=dataset,
    )
    roberta.cuda()
    roberta.eval()

    roberta.load_data(subset)
    output, target = roberta.inference(split=subset, classification_head_name="molecule_head")
    if not isinstance(target, torch.cuda.FloatTensor):
        output = F.softmax(output, dim=-1)
        labels = target.tolist()
        y_pred = output.argmax(dim=-1).tolist()
        scores = output[:, 1].tolist()

        fpr, tpr, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr, tpr)
        auprc = average_precision_score(labels, scores)
        acc = accuracy_score(labels, y_pred, normalize=True)
        print("dataset: {} auroc: {} auprc: {} acc: {}".format(dataset, auroc, auprc, acc))
        scores = [floor(x + 0.5) for x in scores]
        scores_counter = Counter(scores)
        print(scores_counter)
    else:
        y_pred = output.view(-1).tolist()
        labels = target.tolist()
        rmse = ret_rmse(labels, y_pred)
        mae = mean_absolute_error(labels, y_pred)
        print("dataset: {} rmse: {} mae: {}".format(dataset, rmse, mae))
        print(max(labels), min(labels), max(y_pred), min(y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("cktpath", type=str)
    parser.add_argument("--subset", type=str, default="test")
    args = parser.parse_args()
    dataset = args.dataset
    cktpath = args.cktpath
    subset = args.subset
    assert os.path.exists(cktpath)
    main(dataset, cktpath, subset)
