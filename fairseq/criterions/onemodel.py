from fairseq.tasks.onemodel import OneModel
import math
from dataclasses import dataclass, field

from torch.nn.modules import module
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, fairseq_criterion, register_criterion
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Any
from fairseq import metrics, utils, modules
from fairseq.models.onemodel import OneModelConfig
from omegaconf import II
from molecule.features import get_mask_atom_typeid
from torch_geometric.data import Data


@dataclass
class OneModelCriterionConfig(FairseqDataclass):
    datatype: str = II("model.datatype")
    tpu: bool = II("common.tpu")


@register_criterion("onemodel", dataclass=OneModelCriterionConfig)
class OneModelCriterion(FairseqCriterion):
    def __init__(self, task, datatype, tpu):
        super().__init__(task)
        self.datatype = datatype
        self.tpu = tpu
        self.graph_mask_idx = get_mask_atom_typeid()

    def forward(self, model, sample, reduce=True):
        loss, sample_size = self.compute_loss(model, sample, reduce=reduce)
        logging_out = {
            "loss": loss if self.tpu else loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample['nsentences'],
            "sample_size": sample_size
        }
        return loss, sample_size, logging_out

    def update_masked_tokens(self, input, target):
        if 'graph' in input:
            masked_tokens = target.ne(self.graph_mask_idx)
        else:
            masked_tokens = target.ne(self.padding_idx)

        if self.tpu:
            masked_tokens = None
        elif masked_tokens.device == torch.device("cpu"):
            if not masked_tokens.any():
                masked_tokens = None
            else:
                masked_tokens = torch.where(masked_tokens.any(), masked_tokens, masked_tokens.new([True]))
        return masked_tokens

    def compute_loss(self, model: OneModel, sample, reduce=True):
        sample['net_input']['masked_tokens'] = self.update_masked_tokens(sample['net_input'], sample['target'])
        net_output = model(
            net_input=sample['net_input'],
            features_only=False,
        )
        x, net_output_dict = net_output

        targets = model.get_targets(sample['target'], sample['net_input'])
        logits = net_output_dict['pred']
        targets = targets[sample['net_input']['masked_tokens']]
        loss = modules.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='sum')
        sample_size = sample['net_input']['masked_tokens'].int().sum()
        return loss, utils.item(sample_size)

    @staticmethod
    def reduce_metrics(logging_outputs):
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size / math.log(2), sample_size, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
