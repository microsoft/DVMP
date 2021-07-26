from fairseq.models.fairseq_model import FairseqLanguageModel
import logging
from dataclasses import dataclass, field
from typing import Optional

from numpy.lib.arraysetops import isin
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
from fairseq.models.roberta.model import RobertaClassificationHead
from fairseq.models.doublemodel import DoubleModel
from fairseq.modules import GradMultiply


logger = logging.getLogger(__name__)


@dataclass
class OneModelConfig(FairseqDataclass):
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

    gnn_number_layer: int = field(default=12)
    gnn_dropout: float = field(default=0.1)
    conv_encode_edge: bool = field(default=True)
    gnn_embed_dim: int = field(default=384)
    gnn_aggr: str = field(default='maxminmean')
    gnn_norm: str = field(default='batch')
    gnn_activation_fn: str = field(default='relu')

    datatype: str = field(default='g')
    classification_head_name: str = field(default='')
    load_checkpoint_heads: bool = field(default=False)
    gradmultiply: bool = field(default=False)

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


@register_model("onemodel", dataclass=OneModelConfig)
class OneModel(BaseFairseqModel):
    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, args, task):
        if args.datatype == 'g':
            encoder = DeeperGCN(args)
        elif args.datatype == 't':
            from fairseq.models.doublemodel import TrEncoder
            encoder = TrEncoder(args, task.source_dictionary)
        else:
            raise NotImplementedError()
        return cls(args, encoder)

    def forward(self, net_input, features_only=False, classification_head_name=None, **kwargs):
        if classification_head_name is not None:
            features_only = True
        x, pred = self.encoder(**net_input, features_only=features_only, **kwargs)
        if classification_head_name is not None:
            x = self.get_cls(x)
            if self.args.gradmultiply:
                x = GradMultiply.apply(x, 0.1)
            x = self.classification_heads[classification_head_name](x)
            return x

        output_dict = {}
        output_dict['pred'] = pred
        return self.get_cls(x), output_dict

    def get_cls(self, x):
        if isinstance(x, torch.Tensor):
            return x[:, -1, :]
        elif isinstance(x, (tuple, list)):
            return x[0]
        else:
            raise ValueError

    def max_positions(self):
        return self.args.max_positions

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
        input_dim = getattr(self.encoder, "output_features", self.args.encoder_embed_dim)
        self.classification_heads[name] = ClassificationHead(
            input_dim=input_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            activation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
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
        return GraphHubInterface(x["args"], x["task"], x["models"][0])


class ClassificationHead(RobertaClassificationHead):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class GraphHubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))
        self.is_doublemodel = isinstance(self.model, DoubleModel)

    def load_data(self, split='test'):
        self.task.load_dataset(split)

    def inference(self, bsz=16, split='test', classification_head_name='classification_head_name'):
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
            with torch.no_grad():
                if not self.is_doublemodel:
                    pred = self.model(
                        net_input=sample['net_input'],
                        features_only=True,
                        classification_head_name=classification_head_name)
                else:
                    if 'net_input' in sample:
                        pred = self.model(
                            net_input0=sample['net_input'],
                            net_input1=sample['net_input'],
                            features_only=True,
                            classification_head_name=classification_head_name
                        )
                    else:
                        pred = self.model(
                            net_input0=sample['net_input0'],
                            net_input1=sample['net_input1'],
                            features_only=True,
                            classification_head_name=classification_head_name
                        )
            target = self.model.get_targets(sample['target'], None).view(-1)

            preds.append(pred.detach())
            targets.append(target)
        x, y = torch.cat(preds, dim=0), torch.cat(targets, dim=0)

        x = self.task.inverse_transform(x)
        y = self.task.inverse_transform(y)

        return x, y

    @property
    def device(self):
        return self._float_tensor.device
