# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import math
from dataclasses import dataclass, field

from torch.nn.modules import module
from fairseq.dataclass import FairseqDataclass
from fairseq.criterions import FairseqCriterion, register_criterion
import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Dict, Any
from fairseq import metrics, utils, modules
from fairseq.models.doublemodel import DoubleModel
from omegaconf import II
from molecule.features import get_mask_atom_typeid


@dataclass
class DoubleModelCriterionConfig(FairseqDataclass):
    contrastive_t: float = field(default=0.1)
    use_contrastive: bool = field(default=False)
    contrastive_coeff: float = field(default=0.05)
    use_mlm: bool = field(default=False)
    datatype: str = II("model.datatype")
    tpu: bool = II("common.tpu")


@register_criterion("dmp", dataclass=DoubleModelCriterionConfig)
class DoubleModelCriterion(FairseqCriterion):
    def __init__(self, task, contrastive_t, use_contrastive, contrastive_coeff, use_mlm, datatype,
                 tpu):
        super().__init__(task)
        self.contrastive_t = contrastive_t
        self.contrastive_loss = nn.CrossEntropyLoss(reduction='sum')
        self.use_contrastive = use_contrastive
        self.contrastive_coeff = contrastive_coeff
        self.use_mlm = use_mlm
        self.datatype = datatype
        self.tpu = tpu
        self.graph_mask_idx = get_mask_atom_typeid()

    def forward(self, model, sample, reduce=True):

        loss, byol_loss, output_dict = \
            self.forward_tt(model, sample, reduce=reduce)
        sample_size = sample["nsentences"]
        logging_output = {
            "loss": loss.data,
            "byol_loss": byol_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size
        }
        logging_output.update(output_dict)
        return loss, sample_size, logging_output

    def update_masked_tokens(self, input, target):
        if self.use_mlm:
            if "graph" in input:
                masked_tokens = target.ne(self.graph_mask_idx)
            else:
                masked_tokens = target.ne(self.padding_idx)
            if self.tpu:
                masked_tokens = None
            elif masked_tokens.device == torch.device("cpu"):
                if not masked_tokens.any():
                    masked_tokens = None
            else:
                masked_tokens = torch.where(masked_tokens.any(), masked_tokens,
                                            masked_tokens.new([True]))
            return masked_tokens
        else:
            return None

    def forward_tt(self, model: DoubleModel, sample, reduce=True):

        sample['net_input0']['masked_tokens'] = self.update_masked_tokens(
            sample['net_input0'], sample['target0'])
        sample['net_input1']['masked_tokens'] = self.update_masked_tokens(
            sample['net_input1'], sample['target1'])
        return self.compute_loss(model, sample, reduce=reduce)

    def compute_loss(self, model: DoubleModel, sample, reduce=True):
        net_output = model(net_input0=sample['net_input0'],
                           net_input1=sample['net_input1'],
                           features_only=False if self.use_mlm else True,
                           ret_contrastive=True)
        x0, x1, net_output_dict = net_output

        byol_loss = 0
        contrastive_loss = 0
        n_correct, total = 0, 0
        output_dict = {}
        loss = 0

        for i in range(2):
            loss_output = self.get_contrastive_logits(*net_output_dict['contrastive'][i])
            byol_loss = byol_loss + loss_output[0][0]

            if loss_output[0][1] is not None:
                contrastive_loss = contrastive_loss + loss_output[0][1]

            n, t = self.compute_accuracy(*loss_output[1:])
            n_correct += n
            total += t
        loss = loss + byol_loss + contrastive_loss * self.contrastive_coeff
        if isinstance(contrastive_loss, torch.Tensor):
            output_dict["contrastive_loss"] = utils.item(contrastive_loss)

        output_dict["n_correct"] = utils.item(n_correct)
        output_dict["total"] = utils.item(total)

        output_dict["x0_std"] = utils.item(F.normalize(x0, dim=-1, p=2).std(0).mean())
        output_dict["x1_std"] = utils.item(F.normalize(x1, dim=-1, p=2).std(0).mean())
        output_dict["cls_num"] = 1

        if self.use_mlm:
            for i in range(2):
                targets = model.get_targets(sample['target{}'.format(i)],
                                            sample['net_input{}'.format(i)])
                logits = net_output_dict['pred{}'.format(i)]
                targets = targets[sample['net_input{}'.format(i)]['masked_tokens']]
                mlm_loss = modules.cross_entropy(logits.view(-1, logits.size(-1)),
                                                 targets.view(-1),
                                                 reduction='mean')
                output_dict['mlm_loss_{}'.format(i)] = utils.item(mlm_loss)
                loss = loss + sample['nsentences'] * mlm_loss

        return loss, byol_loss, output_dict

    def get_contrastive_logits(self, anchor, positive):
        anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
        positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0]
        logits = logits / self.contrastive_t
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        loss = torch.einsum("nc,nc->n", [anchor, positive])
        loss = -loss.sum()
        if self.use_contrastive:
            c_loss = self.contrastive_loss(logits, targets)
        else:
            c_loss = None
        return (loss, c_loss), logits, targets

    def compute_accuracy(self, logits, targets):
        n_correct = torch.sum(logits.argmax(1).eq(targets))
        total = targets.size(0)
        return utils.item(n_correct), total

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        byol_loss_sum = sum(log.get("byol_loss") for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        tr_cls_std = sum(log.get("x0_std", 0) for log in logging_outputs)
        gnn_cls_std = sum(log.get("x1_std", 0) for log in logging_outputs)
        cls_num = sum(log.get("cls_num", 0) for log in logging_outputs)

        metrics.log_scalar("loss", utils.item(loss_sum) / sample_size, sample_size, round=3)
        metrics.log_scalar("byol_loss",
                           utils.item(byol_loss_sum) / sample_size,
                           sample_size,
                           round=3)
        if 'contrastive_loss' in logging_outputs[0]:
            contrastive_loss_sum = sum(log.get('contrastive_loss', 0) for log in logging_outputs)
            metrics.log_scalar("contrastive_loss",
                               utils.item(contrastive_loss_sum) / sample_size,
                               sample_size,
                               round=3)
        metrics.log_scalar("x0_std", tr_cls_std / cls_num, sample_size, round=3)
        metrics.log_scalar("x1_std", gnn_cls_std / cls_num, sample_size, round=3)

        if 'total' in logging_outputs[0]:
            total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
            metrics.log_scalar("total", total)
            n_correct = utils.item(sum(log.get("n_correct", 0) for log in logging_outputs))
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(meters["n_correct"].sum * 100.0 / meters["total"].sum, 3)
                if meters["total"].sum > 0 else float("nan"),
            )

        for i in range(2):
            key = "mlm_loss_{}".format(i)
            if key in logging_outputs[0]:
                mlm_loss_sum = sum(log.get(key, 0) for log in logging_outputs)
                metrics.log_scalar(key, mlm_loss_sum / cls_num / math.log(2), sample_size, round=3)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True
