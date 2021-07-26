# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
import math
import torch.nn.functional as F
from typing import List, Dict, Any
from fairseq import metrics, utils
from omegaconf import II
from fairseq.models.doublemodel import DoubleModel
from fairseq.models.onemodel import OneModel


@dataclass
class GraphSentencePredictionCriterionConfig(FairseqDataclass):
    classification_head_name: str = II("model.classification_head_name")
    regression_target: bool = II("task.regression_target")
    datatype: str = II("model.datatype")


@register_criterion("graph_sp", dataclass=GraphSentencePredictionCriterionConfig)
class GraphSentencePredictionCriterion(FairseqCriterion):
    def __init__(self, task, classification_head_name, regression_target, datatype):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target
        self.datatype = datatype

    def forward(self, model, sample, reduce=True):
        assert (hasattr(model, 'classification_heads')
                and self.classification_head_name in model.classification_heads)
        if isinstance(model, OneModel):
            logits = model(net_input=sample['net_input'],
                           features_only=True,
                           classification_head_name=self.classification_head_name)
        elif isinstance(model, DoubleModel):
            if 'net_input' in sample:
                logits = model(
                    net_input0=sample['net_input'],
                    net_input1=sample['net_input'],
                    features_only=True,
                    classification_head_name=self.classification_head_name
                )
            else:
                assert "net_input0" in sample
                logits = model(
                    net_input0=sample['net_input0'],
                    net_input1=sample['net_input1'],
                    features_only=True,
                    classification_head_name=self.classification_head_name
                )
        else:
            raise NotImplementedError()
        
        targets = model.get_targets(sample['target'], None).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduce='sum')
        else:
            logits = logits.view(-1).float()
            targets = targets.float()
            loss = F.mse_loss(logits, targets, reduce="sum")

        logging_out = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample_size,
            "sample_size": sample_size
        }
        if not self.regression_target:
            preds = logits.argmax(dim=1)
            logging_out["ncorrect"] = (preds == targets).sum()

        return loss, sample_size, logging_out

    @staticmethod
    def reduce_metrics(logging_outputs):

        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar("accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
