# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
from dataclasses import dataclass, field
from typing import Optional

from numpy.lib.arraysetops import isin
from torch.functional import Tensor
from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (BaseFairseqModel, register_model, register_model_architecture)
import torch
import torch.nn.functional as F
from torch import nn
from fairseq.models.roberta import RobertaEncoder
from fairseq.modules.gnn import GNN
from omegaconf import II
from fairseq.modules import LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_
from fairseq.utils import move_to_cuda
from numpy.random import uniform
from torch_geometric.data import Data
from fairseq.models.gnn import DeeperGCN
from fairseq.modules import GradMultiply

logger = logging.getLogger(__name__)


@dataclass
class DoubleModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(default='gelu', )
    dropout: float = field(default=0.1)
    attention_dropout: float = field(default=0.0)
    activation_dropout: float = field(default=0.0)
    relu_dropout: float = field(default=0.0)
    encoder_embed_path: Optional[str] = field(default=None)
    encoder_embed_dim: int = field(default=768)
    encoder_ffn_embed_dim: int = field(default=3072)
    encoder_layers: int = field(default=12)
    encoder_attention_heads: int = field(default=12)
    encoder_normalize_before: bool = field(default=False)
    encoder_learned_pos: bool = field(default=True)
    layernorm_embedding: bool = field(default=True)
    no_scale_embedding: bool = field(default=True)
    max_positions: int = field(default=512)
    coeff0: float = field(default=1.)
    coeff1: float = field(default=1.)
    use_dropnet: float = field(default=0.)
    datatype: str = field(default='tg')
    gradmultiply: bool = field(default=False)
    gradmultiply_gnn: float = field(default=1.)

    gnn_number_layer: int = field(default=12)
    gnn_dropout: float = field(default=0.1)
    conv_encode_edge: bool = field(default=True)
    gnn_embed_dim: int = field(default=384)
    gnn_aggr: str = field(default='maxminmean')
    gnn_norm: str = field(default='batch')
    gnn_activation_fn: str = field(default='relu')

    classification_head_name: str = field(default='')
    load_checkpoint_heads: bool = field(default=False)

    # config for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
    quant_noise_pq: float = field(
        default=0.0,
        metadata={"help": "iterative PQ quantization noise at training time"},
    )
    quant_noise_pq_block_size: int = field(
        default=8,
        metadata={"help": "block size of quantization noise at training time"},
    )
    quant_noise_scalar: float = field(
        default=0.0,
        metadata={"help": "scalar quantization noise and scalar quantization at training time"},
    )
    # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
    spectral_norm_classification_head: bool = field(default=False)

    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(default=0.0,
                                     metadata={"help": "LayerDrop probability for decoder"})
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={"help": "which layers to *keep* when pruning as a comma-separated list"},
    )
    max_source_positions: int = II("model.max_positions")
    no_token_positional_embeddings: bool = field(default=False)
    pooler_activation_fn: str = field(default='tanh')
    pooler_dropout: float = field(default=0.0)
    untie_weights_roberta: bool = field(default=False)
    adaptive_input: bool = field(default=False)

    head_dropout: float = field(default=0)
    use_bottleneck_head: bool = field(default=False)
    bottleneck_ratio: float = field(default=1)
    bottleneck_layer: int = field(default=2)


@register_model("dmp", dataclass=DoubleModelConfig)
class DoubleModel(BaseFairseqModel):
    def __init__(self, args, encoder0, encoder1):
        super().__init__()
        self.args = args
        self.encoder0 = encoder0
        self.encoder1 = encoder1
        self.classification_heads = nn.ModuleDict()
        self.build_heads()
        self.encoder0_ison = self.args.coeff0 > 0
        self.encoder1_ison = self.args.coeff1 > 0
        self.gradmultiply_gnn = getattr(self.args, "gradmultiply_gnn", 1.0)

    @classmethod
    def build_model(cls, args, task):
        if args.datatype == 'tt':
            encoder0 = TrEncoder(args, task.source_dictionary)
            encoder1 = TrEncoder(args, task.source_dictionary)
        elif args.datatype == 'tg' or args.datatype == 'gt':
            encoder0 = TrEncoder(args, task.source_dictionary)
            encoder1 = DeeperGCN(args)
        elif args.datatype == 'gg':
            encoder0 = DeeperGCN(args)
            encoder1 = DeeperGCN(args)
        else:
            raise NotImplementedError()

        return cls(args, encoder0, encoder1)

    def build_heads(self):
        self.projection_heads = nn.ModuleList()
        if not self.args.use_bottleneck_head:
            self.projection_heads.append(
                NonLinear(getattr(self.encoder0, "output_features", self.args.encoder_embed_dim),
                          ffn_dim=self.args.encoder_embed_dim * 4,
                          out_dim=self.args.encoder_embed_dim,
                          dropout=self.args.head_dropout))
            self.projection_heads.append(
                NonLinear(getattr(self.encoder1, "output_features", self.args.encoder_embed_dim),
                          ffn_dim=self.args.encoder_embed_dim * 4,
                          out_dim=self.args.encoder_embed_dim,
                          dropout=self.args.head_dropout))
            self.prediction_heads = nn.ModuleList(
                [NonLinear(self.args.encoder_embed_dim) for _ in range(2)])
        else:
            self.projection_heads.append(
                BottleNeck(getattr(self.encoder0, "output_features", self.args.encoder_embed_dim),
                           ffn_dim=int(self.args.encoder_embed_dim * self.args.bottleneck_ratio),
                           out_dim=self.args.encoder_embed_dim,
                           mlp_layers=self.args.bottleneck_layer))
            self.projection_heads.append(
                BottleNeck(getattr(self.encoder1, "output_features", self.args.encoder_embed_dim),
                           ffn_dim=int(self.args.encoder_embed_dim * self.args.bottleneck_ratio),
                           out_dim=self.args.encoder_embed_dim,
                           mlp_layers=self.args.bottleneck_layer))
            self.prediction_heads = nn.ModuleList([
                BottleNeck(self.args.encoder_embed_dim,
                           ffn_dim=int(self.args.encoder_embed_dim * self.args.bottleneck_ratio),
                           out_dim=self.args.encoder_embed_dim,
                           mlp_layers=2) for _ in range(2)
            ])

    def forward(self,
                net_input0,
                net_input1,
                features_only=False,
                classification_head_name=None,
                ret_contrastive=False,
                **kwargs):
        if classification_head_name is not None:
            features_only = True

        if self.encoder0_ison:
            x0, pred0 = self.encoder0(**net_input0, features_only=features_only, **kwargs)
        else:
            x0 = None
        if self.encoder1_ison:
            x1, pred1 = self.encoder1(**net_input1, features_only=features_only, **kwargs)
        else:
            x1 = None

        if classification_head_name is not None:
            x0 = self.get_cls(x0)
            x1 = self.get_cls(x1)
            if self.args.gradmultiply:
                if isinstance(x0, Tensor):
                    x0 = GradMultiply.apply(x0, 0.1)
                if isinstance(x1, Tensor):
                    x1 = GradMultiply.apply(x1, 0.1 * self.gradmultiply_gnn)
            x = self.classification_heads[classification_head_name](x0, x1)
            return x

        output_dict = {}
        output_dict['pred0'] = pred0
        output_dict['pred1'] = pred1
        if ret_contrastive:
            output_dict['contrastive'] = \
                [self.get_anchor_positive(x0, x1, first=True)]
            output_dict['contrastive'].append(self.get_anchor_positive(x0, x1, first=False))

        return self.get_cls(x0), self.get_cls(x1), output_dict

    def forward_retrosys(self, net_input):
        return self.encoder0.forward_retrosys(net_input)

    def get_anchor_positive(self, input0, input1, first=True):
        input0 = self.get_cls(input0)
        input1 = self.get_cls(input1)
        if first:
            input0 = self.projection_heads[0](input0)
            input0 = self.prediction_heads[0](input0)
            with torch.no_grad():
                input1 = self.projection_heads[1](input1)
            return input0, input1
        else:
            input1 = self.projection_heads[1](input1)
            input1 = self.prediction_heads[1](input1)
            with torch.no_grad():
                input0 = self.projection_heads[0](input0)
            return input1, input0

    def get_cls(self, x):
        if x is None:
            return 0
        if isinstance(x, torch.Tensor):
            return x[:, -1, :]
        elif isinstance(x, tuple):
            return x[0]
        else:
            raise ValueError()

    def get_targets(self, target, input):
        return target

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning('re-registering head "{}" with num_classes {} (prev: {}) '
                               "and inner_dim {} (prev: {})".format(name, num_classes,
                                                                    prev_num_classes, inner_dim,
                                                                    prev_inner_dim))
        self.classification_heads[name] = TwoInputClassificationHead(
            input0_dim=getattr(self.encoder0, "output_features", self.args.encoder_embed_dim),
            input1_dim=getattr(self.encoder1, "output_features", self.args.encoder_embed_dim),
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            actionvation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
            coeff0=self.args.coeff0,
            coeff1=self.args.coeff1,
            use_dropnet=self.args.use_dropnet,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != "" else ""

        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = ([] if not hasattr(self, "classification_heads") else
                              self.classification_heads.keys())
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[prefix + "classification_heads." + head_name +
                                     ".out_proj.weight"].size(0)
            inner_dim = state_dict[prefix + "classification_heads." + head_name +
                                   ".dense.weight"].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning("deleting classification head ({}) from checkpoint "
                                   "not present in current model: {}".format(head_name, k))
                    keys_to_delete.append(k)
                elif (num_classes != self.classification_heads[head_name].out_proj.out_features
                      or inner_dim != self.classification_heads[head_name].dense.out_features):
                    logger.warning("deleting classification head ({}) from checkpoint "
                                   "with different dimensions than current model: {}".format(
                                       head_name, k))
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if hasattr(self, "classification_heads"):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

    def max_positions(self):
        return self.args.max_positions

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        **kwargs,
    ):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(model_name_or_path,
                                      checkpoint_file,
                                      data_name_or_path,
                                      load_checkpoint_heads=True,
                                      **kwargs)
        logger.info(x["args"])
        return TwoModelHubInterface(x["args"], x["task"], x["models"][0])


class TrEncoder(RobertaEncoder):
    def __init(self, **kwargs):
        super().__init__(**kwargs)

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        features, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)
        if not features_only:
            x = self.output_layer(features, masked_tokens=masked_tokens)
        else:
            x = None
        return features, x

    def forward_retrosys(self, src_tokens):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=False,
            token_embeddings=None
        )
        encoder_out_new = dict()
        encoder_out_new["encoder_out"] = encoder_out["encoder_out"]
        encoder_out_new["encoder_padding_mask"] = encoder_out["encoder_padding_mask"]
        return encoder_out_new


class TwoModelHubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    def load_data(self, split='test'):
        self.task.load_dataset(split)

    def inference(self, bsz=8, split='test', classification_head_name='classification_head_name'):
        dataset = self.task.datasets[split]
        total = len(dataset)

        preds = []
        targets = []
        for i in range(0, total, bsz):
            data_list = []
            for j in range(0, bsz):
                if i + j >= total:
                    break
                data_list.append(dataset[i + j])

            sample = dataset.collater(data_list)
            sample = move_to_cuda(sample, self.device)
            pred = self.model(graph_data=sample['graph'],
                              **sample['net_input'],
                              features_only=True,
                              classification_head_name=classification_head_name).detach()
            target = self.model.get_targets(sample, [pred]).view(-1)

            preds.append(pred)
            targets.append(target)

        return torch.cat(preds, dim=0), torch.cat(targets, dim=0)

    @property
    def device(self):
        return self._float_tensor.device


class NonLinear(nn.Module):
    def __init__(self, in_dim, ffn_dim=None, out_dim=None, dropout=0.):
        super().__init__()

        ffn_dim = ffn_dim if ffn_dim is not None else in_dim * 4
        self.fc1 = nn.Linear(in_dim, ffn_dim)
        out_dim = out_dim if out_dim is not None else in_dim
        self.fc2 = nn.Linear(ffn_dim, out_dim)
        self.ln = LayerNorm(out_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, self.dropout)
        x = self.ln(x)
        return x


class BottleNeck(nn.Module):
    def __init__(self, in_dim, ffn_dim=None, out_dim=None, mlp_layers=2):
        super().__init__()
        ffn_dim = ffn_dim if ffn_dim is not None else in_dim
        out_dim = out_dim if out_dim is not None else in_dim

        self.mlps = nn.ModuleList()
        self.mlps.append(nn.Linear(in_dim, ffn_dim))
        for _ in range(mlp_layers - 2):
            self.mlps.append(nn.Linear(ffn_dim, ffn_dim))
        self.mlps.append(nn.Linear(ffn_dim, out_dim))

        self.bns = nn.ModuleList()
        for _ in range(mlp_layers - 1):
            self.bns.append(nn.BatchNorm1d(ffn_dim))

        self.num_layers = mlp_layers

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.mlps[i](x)
            x = self.bns[i](x)
            x = F.relu(x)
        x = self.mlps[-1](x)
        return x


class TwoInputClassificationHead(nn.Module):
    def __init__(self,
                 input0_dim,
                 input1_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout,
                 q_noise=0,
                 qn_block_size=8,
                 do_spectral_norm=False,
                 coeff0=1.,
                 coeff1=1.,
                 use_dropnet=False):
        super().__init__()
        self.dense = nn.Linear(input0_dim, inner_dim)
        self.dense1 = nn.Linear(input1_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(nn.Linear(inner_dim, num_classes), q_noise,
                                           qn_block_size)
        self.coeff0 = coeff0
        self.coeff1 = coeff1
        self.use_dropnet = use_dropnet
        
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError()
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, input0, input1):
        if self.coeff0 > 0:
            input0 = self.dropout(input0)
            input0 = self.dense(input0)
            input0 = self.activation_fn(input0)
            input0 = self.dropout(input0)
            input0 = self.out_proj(input0)
        if self.coeff1 > 0:
            input1 = self.dropout(input1)
            input1 = self.dense1(input1)
            input1 = self.activation_fn(input1)
            input1 = self.dropout(input1)
            input1 = self.out_proj(input1)

        dropnet = self.get_dropnet()
        x = dropnet[0] * input0 + dropnet[1] * input1
        return x

    def get_dropnet(self):
        if self.training and self.use_dropnet > 0:
            frand = float(uniform(0, 1))
            if frand < self.use_dropnet:
                return (self.coeff0, 0)
            elif frand > 1 - self.use_dropnet:
                return (0, self.coeff1)
            else:
                return (self.coeff0, self.coeff1)
        else:
            return (self.coeff0, self.coeff1)
