import logging
from dataclasses import dataclass, field
from typing import Optional
from fairseq import options, utils
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.models import (BaseFairseqModel,
                            register_model,
                            register_model_architecture)
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


logger = logging.getLogger(__name__)


@dataclass
class GraphSeqModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default='gelu',
    )
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
    freeze_bn: bool = field(default=False)
    graph_coff: float = field(default=1.)
    seq_coff: float = field(default=1.)
    use_dropnet: bool = field(default=False)

    classification_head_name: str = field(default='')
    load_checkpoint_heads: bool = field(default=False)

    gnn_type: str = field(default='gin')
    gnn_number_layer: int = field(default=6)
    gnn_embed_dim: int = field(default=768)
    gnn_JK: str = field(default='last')
    gnn_dropout: float = field(default=0.)
    gnn_pooling: str = field(default='mean')

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
        metadata={
            "help": "scalar quantization noise and scalar quantization at training time"
        },
    )
    # args for "Better Fine-Tuning by Reducing Representational Collapse" (Aghajanyan et al. 2020)
    spectral_norm_classification_head: bool = field(default=False)

    # config for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
    encoder_layerdrop: float = field(
        default=0.0, metadata={"help": "LayerDrop probability for decoder"}
    )
    encoder_layers_to_keep: Optional[str] = field(
        default=None,
        metadata={
            "help": "which layers to *keep* when pruning as a comma-separated list"
        },
    )
    max_source_positions: int = II("model.max_positions")
    no_token_positional_embeddings: bool = field(default=False)
    pooler_activation_fn: str = field(default='tanh')
    pooler_dropout: float = field(default=0.0)
    untie_weights_roberta: bool = field(default=False)
    adaptive_input: bool = field(default=False)


@register_model("graphseq", dataclass=GraphSeqModelConfig)
class GraphSeqModel(BaseFairseqModel):

    def __init__(self, args, tr_encoder, gnn_encoder):
        super().__init__()
        self.args = args
        self.tr_encoder = tr_encoder
        self.gnn_encoder = gnn_encoder
        self.classification_heads = nn.ModuleDict()
        self.build_heads()

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        tr_encoder = TrEncoder(args, task.src_dict)
        gnn_encoder = GNN(num_layer=args.gnn_number_layer,
                          emb_dim=args.gnn_embed_dim,
                          JK=args.gnn_JK,
                          dropout=args.gnn_dropout,
                          gnn_type=args.gnn_type,
                          graph_pooling=args.gnn_pooling,
                          freeze_bn=args.freeze_bn
                          )
        return cls(args, tr_encoder, gnn_encoder)

    def build_heads(self):
        self.projection_heads = nn.ModuleList(
            [NonLinear(self.args.encoder_embed_dim) for _ in range(2)]
        )
        self.prediction_heads = nn.ModuleList(
            [NonLinear(self.args.encoder_embed_dim) for _ in range(2)]
        )

    def forward(self,
                src_tokens,
                graph_data,
                features_only=False,
                return_all_hiddens=False,
                classification_head_name=None,
                ret_contrastive=False,
                **kwargs):
        if classification_head_name is not None:
            features_only = True
    
        x_tr, pred_seqs = self.tr_encoder(src_tokens, features_only, return_all_hiddens,
                                   **kwargs)
        x_gnn = self.gnn_encoder(graph_data)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x_tr, x_gnn)
            return x

        output_dict = {}
        if pred_seqs is not None:
            output_dict['pred_seqs'] = pred_seqs
        if ret_contrastive:
            output_dict['contrasitve'] = \
                [self.get_anchor_positive(x_tr, x_gnn, seq_prediction=True)]
            output_dict['contrasitve'].append(
                self.get_anchor_positive(x_tr, x_gnn, seq_prediction=False)
            )

        return x_tr, x_gnn, output_dict

    def get_anchor_positive(self, seq, graph, seq_prediction=True):
        seq = seq[:, -1, :]
        graph = graph[0]
        if seq_prediction:
            seq_anchor = self.projection_heads[0](seq)
            seq_anchor = self.prediction_heads[0](seq_anchor)
            with torch.no_grad():
                graph_positive = self.projection_heads[1](graph)
            return seq_anchor, graph_positive
        else:
            graph_anchor = self.projection_heads[1](graph)
            graph_anchor = self.prediction_heads[1](graph_anchor)
            with torch.no_grad():
                seq_positive = self.projection_heads[0](seq)
            return graph_anchor, seq_positive


    def register_classification_head(self, name, num_classes=None,
                                     inner_dim=None, **kwargs):
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = TwoInputClassificationHead(
            input_dim=self.args.encoder_embed_dim,
            inner_dim=inner_dim or self.args.encoder_embed_dim,
            num_classes=num_classes,
            actionvation_fn=self.args.pooler_activation_fn,
            pooler_dropout=self.args.pooler_dropout,
            q_noise=self.args.quant_noise_pq,
            qn_block_size=self.args.quant_noise_pq_block_size,
            do_spectral_norm=self.args.spectral_norm_classification_head,
            graph_coff=self.args.graph_coff,
            seq_coff=self.args.seq_coff,
            use_dropnet=self.args.use_dropnet,
        )

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + '.' if name != "" else ""

        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads")
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads."):].split(".")[0]
            num_classes = state_dict[
                prefix + "classification_heads." + head_name + ".out_proj.weight"
                ].size(0)
            inner_dim = state_dict[
                prefix + "classification_heads." + head_name + ".dense.weight"
                ].size(0)

            if getattr(self.args, "load_checkpoint_heads", False):
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                        num_classes
                        != self.classification_heads[head_name].out_proj.out_features
                        or inner_dim
                        != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(
                            head_name, k
                        )
                    )
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
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            load_checkpoint_heads=True,
            **kwargs
        )
        logger.info(x["args"])
        return GraphSeqHUbInterface(x["args"], x["task"], x["models"][0])


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
        features, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(features, masked_tokens=masked_tokens)
        else:
            x = None
        return features, x


class GraphSeqHUbInterface(nn.Module):

    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    def load_data(self, split='test'):
        self.task.load_dataset(split)

    def inference(self, bsz=8, split='test',
                  classification_head_name='classification_head_name'):
        dataset = self.task.datasets[split]
        total = len(dataset)

        preds = []
        targets = []
        for i in range(0, total, bsz):
            data_list = []
            for j in range(0, bsz):
                if i + j >= total:
                    break
                data_list.append(dataset[i+j])

            sample = dataset.collater(data_list)
            sample = move_to_cuda(sample, self.device)
            pred = self.model(
                graph_data=sample['graph'],
                **sample['net_input'],
                features_only=True,
                classification_head_name=classification_head_name
            ).detach()
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
        self.ln = LayerNorm(in_dim)
        self.dropout = dropout

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.dropout(x, self.dropout)
        x = self.ln(x)
        return x


class TwoInputClassificationHead(nn.Module):

    def __init__(self,
                 input_dim,
                 inner_dim,
                 num_classes,
                 actionvation_fn,
                 pooler_dropout,
                 q_noise=0,
                 qn_block_size=8,
                 do_spectral_norm=False,
                 graph_coff=1.,
                 seq_coff=1.,
                 use_dropnet=False):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.ln_graph = LayerNorm(inner_dim)
        self.ln_seq = LayerNorm(inner_dim)
        self.activation_fn = utils.get_activation_fn(actionvation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = apply_quant_noise_(
            nn.Linear(inner_dim, num_classes), q_noise, qn_block_size
        )
        self.grah_coff = graph_coff
        self.seq_coff = seq_coff
        self.use_dropnet = use_dropnet
        if do_spectral_norm:
            if q_noise != 0:
                raise NotImplementedError()
            self.out_proj = torch.nn.utils.spectral_norm(self.out_proj)

    def forward(self, seq, graph):
        seq = seq[:, -1, :]
        graph = graph[0]

        seq = self.ln_seq(seq)
        graph = self.ln_graph(graph)
        seq = self.dropout(seq)
        graph = self.dropout(graph)
        dropnet = self.get_dropnet()
        x = seq * dropnet[0] + graph * dropnet[1]
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

    def get_dropnet(self):
        if self.training and self.use_dropnet:
            if uniform(0, 1) > 0.5:
                return (0, self.grah_coff)
            else:
                return (self.seq_coff, 0)
        else:
            return (self.seq_coff, self.grah_coff)


@register_model_architecture("graphseq", "graphseq_base")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 3072)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.)






