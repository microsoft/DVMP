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
from fairseq.models.graphseq import GraphSeqModel


@dataclass
class ContrastiveCrossEntropyCriterionConfig(FairseqDataclass):
    contrastive_t: float = field(default=0.1)
    normalize_before_ct: bool = field(default=False)
    report_accuracy: bool = field(default=False)
    use_byol_cosine: bool = field(default=False)
    use_mlm: bool = field(default=False)


@register_criterion(
    "contrastive_cross_entropy", dataclass=ContrastiveCrossEntropyCriterionConfig
)
class ContrastiveCrossEntropyCriterion(FairseqCriterion):
    def __init__(self,
                 task,
                 contrastive_t,
                 normalize_before_ct,
                 report_accuracy,
                 use_byol_cosine,
                 use_mlm):
        super().__init__(task)
        self.contrastive_t = contrastive_t
        self.normalize_before_ct = normalize_before_ct
        self.contrastive_loss = nn.CrossEntropyLoss(reduction='sum')
        self.report_accuracy = report_accuracy
        self.use_byol_cosine = use_byol_cosine
        self.use_mlm = use_mlm

    def forward(self, model, sample, reduce=True):
        
        if self.use_mlm:
            masked_tokens = sample['seq_masked_pos']
        else:
            sample['graph'].masked_pos = None
            masked_tokens = None
        

        net_output = model(graph_data=sample['graph'], **sample['net_input'],
                           features_only=False if self.use_mlm else True,
                           ret_contrastive=True,
                           masked_tokens=masked_tokens)
        loss, contrastive_loss, output_dict = self.compute_loss(model, net_output,
                                                   sample, reduce=reduce)
        sample_size = sample["nsentences"]
        logging_output = {
            "loss": loss.data,
            "contrastive_loss": contrastive_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size
        }
        logging_output.update(output_dict)
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        seq, graph, net_output_dict = net_output

        contrastive_loss = 0
        n_correct, total = 0, 0
        output_dict = {}
        loss = 0

        for i in range(2):
            loss_output = self.get_contrastive_logits(*net_output_dict['contrasitve'][i])
            contrastive_loss = contrastive_loss + loss_output[0]

            if self.report_accuracy:
                n, t = self.compute_accuracy(*loss_output[1:])
                n_correct += n
                total += t

        if self.report_accuracy:
            output_dict["n_correct"] = utils.item(n_correct)
            output_dict["total"] = utils.item(total)

        output_dict["tr_cls_std"] = utils.item(
            F.normalize(seq[:, -1, :], dim=-1, p=2).std(0).mean()
        )
        output_dict["gnn_cls_std"] = utils.item(
            F.normalize(graph[0], dim=-1, p=2).std(0).mean()
        )
        output_dict["cls_num"] = 1

        if self.use_mlm:
            # mlm for seq input 
            scale = sample['nsentences'] / sample['ntokens'] / self.task.seq_mask_prob
            target = sample['masked_tokens_label'][sample['seq_masked_pos']]
            logits = net_output_dict['pred_seqs']
            mlm_loss_seq = modules.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                reduction='sum'
            ) * scale
            # mlm for graph input
            scale = sample['nsentences'] / sample['graph'].num_nodes / self.task.graph_mask_prob
            logits = graph[2]
            target = sample["graph"].x_label[:,0][sample['graph'].masked_pos]
            mlm_loss_graph = modules.cross_entropy(
                logits.view(-1, logits.size(-1)),
                target.view(-1),
                reduction='sum'
            ) * scale

            loss = mlm_loss_graph + mlm_loss_seq

            output_dict['mlm_loss_graph'] = utils.item(mlm_loss_graph)
            output_dict['mlm_loss_seq'] = utils.item(mlm_loss_seq)

        loss = contrastive_loss + loss

        return loss, contrastive_loss, output_dict

    def get_contrastive_logits(self, anchor, positive):
        if self.normalize_before_ct:
            anchor = anchor / torch.norm(anchor, dim=-1, keepdim=True)
            positive = positive / torch.norm(positive, dim=-1, keepdim=True)
        logits = torch.matmul(anchor, positive.T)
        logits = logits - torch.max(logits, 1, keepdim=True)[0]
        logits = logits / self.contrastive_t
        targets = torch.arange(logits.shape[1]).long().to(logits.device)
        if self.use_byol_cosine:
            loss = torch.einsum("nc,nc->n", [anchor, positive])
            loss = - loss.sum()
        else:
            loss = self.contrastive_loss(logits, targets)
        return loss, logits, targets

    def compute_accuracy(self, logits, targets):
        n_correct = torch.sum(
            logits.argmax(1).eq(targets)
        )
        total = targets.size(0)
        return utils.item(n_correct), total

    @classmethod
    def reduce_metrics(cls, logging_outputs: List[Dict[str, Any]]) -> None:
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        contrastive_loss_sum = sum(log.get("contrastive_loss") for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        tr_cls_std = sum(log.get("tr_cls_std", 0) for log in logging_outputs)
        gnn_cls_std = sum(log.get("gnn_cls_std", 0) for log in logging_outputs)
        cls_num = sum(log.get("cls_num", 0) for log in logging_outputs)


        metrics.log_scalar(
            "loss", utils.item(loss_sum) / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "contrastive_loss", utils.item(contrastive_loss_sum) / sample_size,
            sample_size, round=3
        )
        metrics.log_scalar(
            "tr_cls_std", tr_cls_std / cls_num, sample_size, round=3
        )
        metrics.log_scalar(
            "gnn_cls_std", gnn_cls_std / cls_num, sample_size, round=3
        )
        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )
        if 'mlm_loss_graph' in logging_outputs[0]:
            mlm_loss_graph_sum = sum(log.get('mlm_loss_graph', 0) for log in logging_outputs)
            mlm_loss_seq_sum = sum(log.get('mlm_loss_seq') for log in logging_outputs)
            metrics.log_scalar(
                "mlm_loss_graph", mlm_loss_graph_sum / sample_size / math.log(2), round=3
            )
            metrics.log_scalar(
                "mlm_loss_seq", mlm_loss_seq_sum / sample_size / math.log(2), round=3
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        return True