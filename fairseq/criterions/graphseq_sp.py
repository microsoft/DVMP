from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
import math
import torch.nn.functional as F
from typing import List, Dict, Any
from fairseq import metrics, utils
from omegaconf import II


@dataclass
class GraphSeqSentencePredictionCriterionConfig(FairseqDataclass):
    classification_head_name: str = II("model.classification_head_name")
    regression_target: bool = II("task.regression_target")


@register_criterion(
    "graphseq_sp", dataclass=GraphSeqSentencePredictionCriterionConfig
)
class GraphSeqSentencePredictionCriterion(FairseqCriterion):

    def __init__(self,
                 task,
                 classification_head_name,
                 regression_target):
        super().__init__(task)
        self.classification_head_name = classification_head_name
        self.regression_target = regression_target

    def forward(self, model, sample, reduce=True):
        assert (
            hasattr(model, 'classification_heads')
            and self.classification_head_name in model.classification_heads
        )
        logits = model(
            graph_data=sample['graph'],
            **sample['net_input'],
            features_only=True,
            classification_head_name=self.classification_head_name
        )
        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.regression_target:
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
            loss = F.nll_loss(lprobs, targets, reduction='sum')
        else:
            raise NotImplementedError

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

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )

        if len(logging_outputs) > 0 and "ncorrect" in logging_outputs[0]:
            ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
            metrics.log_scalar(
                "accuracy", 100.0 * ncorrect / nsentences, nsentences, round=1
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True

