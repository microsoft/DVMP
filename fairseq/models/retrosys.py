from argparse import ArgumentParser, ArgumentTypeError, ArgumentError, Namespace
from dataclasses import dataclass, _MISSING_TYPE, MISSING
from enum import Enum
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.models.transformer import Embedding, TransformerDecoder, TransformerEncoder
from fairseq.dataclass.utils import interpret_dc_type, eval_str_list, gen_parser_from_dataclass
import inspect
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.tasks import register_task
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.distributed import fsdp_wrap
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    BaseLayer,
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from torch import Tensor
from fairseq.models.doublemodel import DoubleModel
from random import uniform
from fairseq.modules import GradMultiply


DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024

DEFAULT_MIN_PARAMS_TO_WRAP = int(1e8)


def gen_plm_parser_from_dataclass(
    parser: ArgumentParser, dataclass_instance: FairseqDataclass, delete_default: bool = False
) -> None:
    def argparse_name(name: str):
        if name == "data":
            return name
        if name == "_name":
            return None
        return "--plm-" + name.replace("_", "-")

    def get_kwargs_from_dc(dataclass_instance: FairseqDataclass, k: str) -> Dict[str, Any]:

        kwargs = {}
        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)
        field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)

        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (
                isinstance(inter_type, type)
                and (issubclass(inter_type, List) or issubclass(inter_type, Tuple))
            ) or ("List" in str(inter_type) or "Tuple" in str(inter_type)):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError(
                        "parsing of type " + str(inter_type) + " is not implemented"
                    )
                if field_default is not MISSING:
                    kwargs["default"] = (
                        ",".join(map(str, field_default)) if field_default is not None else None
                    )
            elif (isinstance(inter_type, type) and issubclass(inter_type, Enum)) or "Enum" in str(
                inter_type
            ):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = "store_false" if field_default is True else "store_true"
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"

        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        field_type = dataclass_instance._get_type(k)
        if field_name is None:
            continue
        elif inspect.isclass(field_type) and issubclass(field_type, FairseqDataclass):
            gen_parser_from_dataclass(parser, field_type(), delete_default)
            continue

        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith("${"):
                if kwargs["help"] is None:
                    # this is a field with a name that will be added elsewhere
                    continue
                else:
                    del kwargs["default"]
            if delete_default and "default" in kwargs:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass


def gen_plm_args(args: Namespace) -> Namespace:
    kwargs = {}
    for k, v in vars(args).items():
        if k.startswith("plm_"):
            kwargs[k[len("plm_") :]] = v
    return Namespace(**kwargs)


@register_model("retrosys")
class RetroSysModel(FairseqEncoderDecoderModel):
    def __init__(self, args, encoder, decoder):
        super().__init__(encoder, decoder)
        self.args = args
        self.supports_align_args = True

    @staticmethod
    def add_args(parser: ArgumentParser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D', help='dropout probability')
        parser.add_argument('--attention-dropout',
                            type=float,
                            metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout',
                            '--relu-dropout',
                            type=float,
                            metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path',
                            type=str,
                            metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim',
                            type=int,
                            metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim',
                            type=int,
                            metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N', help='num encoder layers')
        parser.add_argument('--encoder-attention-heads',
                            type=int,
                            metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before',
                            action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos',
                            action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path',
                            type=str,
                            metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim',
                            type=int,
                            metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim',
                            type=int,
                            metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N', help='num decoder layers')
        parser.add_argument('--decoder-attention-heads',
                            type=int,
                            metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos',
                            action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before',
                            action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--decoder-output-dim',
                            type=int,
                            metavar='N',
                            help='decoder output dimension (extra linear layer '
                            'if different from decoder embed dim')
        parser.add_argument('--share-decoder-input-output-embed',
                            action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings',
                            action='store_true',
                            help='share encoder, decoder and output embeddings'
                            ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings',
                            default=False,
                            action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff',
                            metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                            'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout',
                            type=float,
                            metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--layernorm-embedding',
                            action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding',
                            action='store_true',
                            help='if True, dont scale embeddings')
        parser.add_argument('--checkpoint-activations',
                            action='store_true',
                            help='checkpoint activations at each layer, which saves GPU '
                            'memory usage at the cost of some additional compute')
        parser.add_argument(
            '--offload-activations',
            action='store_true',
            help=
            'checkpoint activations at each layer, then save to gpu. Sets --checkpoint-activations.'
        )
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention',
                            default=False,
                            action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention',
                            default=False,
                            action='store_true',
                            help='perform cross+self-attention')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop',
                            type=float,
                            metavar='D',
                            default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop',
                            type=float,
                            metavar='D',
                            default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep',
                            default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep',
                            default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument('--quant-noise-pq',
                            type=float,
                            metavar='D',
                            default=0,
                            help='iterative PQ quantization noise at training time')
        parser.add_argument('--quant-noise-pq-block-size',
                            type=int,
                            metavar='D',
                            default=8,
                            help='block size of quantization noise at training time')
        parser.add_argument(
            '--quant-noise-scalar',
            type=float,
            metavar='D',
            default=0,
            help='scalar quantization noise and scalar quantization at training time')
        # args for Fully Sharded Data Parallel (FSDP) training
        parser.add_argument(
            '--min-params-to-wrap',
            type=int,
            metavar='D',
            default=DEFAULT_MIN_PARAMS_TO_WRAP,
            help=('minimum number of params for a layer to be wrapped with FSDP() when '
                  'training with --ddp-backend=fully_sharded. Smaller values will '
                  'improve memory efficiency, but may make torch.distributed '
                  'communication less efficient due to smaller input sizes. This option '
                  'is set to 0 (i.e., always wrap) when --checkpoint-activations or '
                  '--offload-activations are passed.'))

        # args for doublemodel
        from fairseq.models.doublemodel import DoubleModelConfig
        gen_plm_parser_from_dataclass(parser, DoubleModelConfig())

        # args for bertnmt
        parser.add_argument(
            "--dropnet",
            type=float,
            default=0
        )
        parser.add_argument(
            "--gradmultiply",
            type=float,
            default=1.
        )
        parser.add_argument(
            "--finetune-plm",
            action="store_true",
            default=False
        )
        parser.add_argument(
            "--plm-grad",
            type=float,
            default=None
        )
        parser.add_argument(
            "--from-scratch",
            action='store_true',
            default=False
        )
        parser.add_argument(
            "--plm-as-encoder",
            action="store_true",
            default=False
        )

        # fmt: on

    def load_state_dict(self, state_dict, strict=True, model_cfg=None, args=None):
        keys_to_delete = []
        cur_state = self.state_dict()

        for k in state_dict.keys():
            if k.startswith("encoder.plm_encoder.projection_heads") or k.startswith(
                "encoder.plm_encoder.prediction_heads"
            ):
                keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]
        for k in cur_state.keys():
            if k.startswith("encoder.plm_encoder.projection_heads") or k.startswith(
                "encoder.plm_encoder.prediction_heads"
            ):
                state_dict[k] = cur_state[k]
            if "plm_attn" in k:
                state_dict[k] = cur_state[k]
            elif self.args.from_scratch and not k.startswith("encoder.plm_encoder"):
                state_dict[k] = cur_state[k]
        super().load_state_dict(state_dict, strict=strict, model_cfg=model_cfg, args=args)

    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError("--share-all-embeddings not compatible with --decoder-embed-path")
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )
        if getattr(args, "offload_activations", False):
            args.checkpoint_activations = True  # offloading implies checkpointing

        args_new = gen_plm_args(args)
        task_new = Namespace(source_dictionary=task.plm_dict)
        plm = DoubleModel.build_model(args_new, task_new)

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, plm)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)

        if not args.share_all_embeddings:
            min_params_to_wrap = getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            # fsdp_wrap is a no-op when --ddp-backend != fully_sharded
            encoder = fsdp_wrap(encoder, min_num_params=min_params_to_wrap)
            decoder = fsdp_wrap(decoder, min_num_params=min_params_to_wrap)
            # plm = fsdp_wrap(plm, min_num_params=min_params_to_wrap)
        return cls(args, encoder, decoder)

    @classmethod
    def build_embedding(cls, args, dictionary, embed_dim, path=None):
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        emb = Embedding(num_embeddings, embed_dim, padding_idx)
        if path:
            embed_dict = utils.parse_embedding(path)
            utils.load_embedding(embed_dict, dictionary, emb)
        return emb

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, plm_encoder):
        return Encoder(args, src_dict, embed_tokens, plm_encoder)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return Decoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    def forward(
        self,
        src_tokens,
        src_lengths,
        prev_output_tokens,
        plm_input,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
            plm_input=plm_input,
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

    @torch.jit.export
    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None,
    ):
        """Get normalized probabilities (or log probs) from a net's output."""
        return self.get_normalized_probs_scriptable(net_output, log_probs, sample)


class Encoder(TransformerEncoder):
    def __init__(self, args, src_dict, embed_tokens, plm_encoder):
        super().__init__(args, src_dict, embed_tokens)
        self.plm_encoder = plm_encoder
        self.finetune_plm = getattr(args, "finetune_plm", False)
        if not self.finetune_plm:
            for p in self.plm_encoder.parameters():
                p.requires_grad = False
        else:
            for n, p in self.plm_encoder.named_parameters():
                if n.startswith("prediction_heads") or n.startswith("projection_heads"):
                    p.requires_grad = False
                if n.startswith("encoder0.lm_head"):
                    p.requires_grad = False
                if n.startswith("encoder1"):
                    p.requires_grad = False
        self.gradmultiply = getattr(args, "plm_grad", None)
        if self.gradmultiply is None:
            self.gradmultiply = 1 / getattr(args, "gradmultiply", 1.)

    def build_encoder_layer(self, args):
        layer = EncoderLayer(args)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP) if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer

    def forward(
        self,
        src_tokens,
        plm_input=None,
        src_lengths: Optional[torch.Tensor] = None,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        plm_input = plm_input["net_input0"]["src_tokens"]
        return self.forward_scriptable(
            src_tokens,
            plm_input,
            src_lengths,
            return_all_hiddens,
            token_embeddings,
        )

    def forward_scriptable(
        self,
        src_tokens,
        plm_input: Optional[torch.Tensor],
        src_lengths: Optional[torch.Tensor],
        return_all_hiddens: bool,
        token_embeddings: Optional[torch.Tensor],
    ):

        plm_out = self.plm_encoder.forward_retrosys(plm_input)
        if self.finetune_plm:
            plm_out["encoder_out"][0] = GradMultiply.apply(plm_out["encoder_out"][0], self.gradmultiply)

        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        has_pads = src_tokens.device.type == "xla" or encoder_padding_mask.any()
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        if has_pads:
            x = x * (1 - encoder_padding_mask.unsqueeze(-1).type_as(x))

            # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states = []

        if return_all_hiddens:
            encoder_states.append(x)

        if plm_out is not None:
            plm_padding_mask = plm_out["encoder_padding_mask"][0]
            plm_out = plm_out["encoder_out"][0]
        plm_has_pads = plm_out.device.type == "xla" or plm_padding_mask.any()

        # encoder layers
        for layer in self.layers:
            x = layer(
                x,
                encoder_padding_mask=encoder_padding_mask if has_pads else None,
                plm_out=plm_out,
                plm_padding_mask=plm_padding_mask if plm_has_pads else None,
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # The Pytorch Mobile lite interpreter does not supports returning NamedTuple in
        # `forward` so we use a dictionary instead.
        # TorchScript does not support mixed values so the values are all lists.
        # The empty list is equivalent to None.
        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "plm_out": [plm_out],  # T x B x C
            "plm_padding_mask": [plm_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        new_encoder_out = super().reorder_encoder_out(encoder_out, new_order)
        if len(encoder_out["plm_out"]) == 0:
            new_plm_out = []
        else:
            new_plm_out = [encoder_out["plm_out"][0].index_select(1, new_order)]
        if len(encoder_out["plm_padding_mask"]) == 0:
            new_plm_paddding_mask = []
        else:
            new_plm_paddding_mask = [encoder_out["plm_padding_mask"][0].index_select(0, new_order)]
        new_encoder_out.update({"plm_out": new_plm_out, "plm_padding_mask": new_plm_paddding_mask})
        return new_encoder_out


class Decoder(TransformerDecoder):
    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]],
        full_context_alignment: bool,
        alignment_layer: Optional[int],
        alignment_heads: Optional[int],
    ):
        bs, slen = prev_output_tokens.size()
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        enc: Optional[Tensor] = None
        padding_mask: Optional[Tensor] = None
        plm: Optional[Tensor] = None
        plm_padding_mask: Optional[Tensor] = None
        if encoder_out is not None:
            enc = encoder_out["encoder_out"][0]
            padding_mask = encoder_out["encoder_padding_mask"][0]
            assert enc.size()[1] == bs, f"Expected enc.shape == (t, {bs}, c) got {enc.shape}"
            plm = encoder_out["plm_out"][0]
            plm_padding_mask = encoder_out["plm_padding_mask"][0]
            assert plm.size()[1] == bs, f"Expected plm.shape == (t, {bs}, c) got {plm.shape}"

        positions = None
        if self.embed_positions is not None:
            positions = self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            x, layer_attn, _ = layer(
                x,
                enc,
                padding_mask,
                plm,
                plm_padding_mask,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=bool((idx == alignment_layer)),
                need_head_weights=bool((idx == alignment_layer)),
            )
            inner_states.append(x)
            if layer_attn is not None and idx == alignment_layer:
                attn = layer_attn.float().to(x)

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states}

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = DecoderLayer(args, no_encoder_attn=no_encoder_attn)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP) if not checkpoint else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class EncoderLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.plm_attn = self.build_encoder_plm_attention(self.embed_dim, args)

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.dropnet = getattr(args, "dropnet", 0.25)
        self.gradmultiply = getattr(args, "gradmultiply", 1.)
        self.plm_as_encoder = getattr(args, "plm_as_encoder", False)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_plm_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            kdim=args.plm_encoder_embed_dim,
            vdim=args.plm_encoder_embed_dim,
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def residual_connection(self, x, residual):
        return residual + x

    def upgrade_state_dict_named(self, state_dict, name):
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def get_ratio(self):
        if self.plm_as_encoder:
            return [0, 1]
        if self.dropnet > 0 and self.training:
            frand = float(uniform(0, 1))
            if frand < self.dropnet:
                return [1, 0]
            elif frand > 1 - self.dropnet:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [0.5, 0.5]

    def forward(
        self,
        x,
        plm_out,
        encoder_padding_mask: Optional[Tensor],
        plm_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)

        x1, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
        )
        x2, _ = self.plm_attn(
            query=x, key=plm_out, value=plm_out, key_padding_mask=plm_padding_mask, attn_mask=None
        )
        x1 = self.dropout_module(x1)
        x2 = self.dropout_module(x2)
        x2 = GradMultiply.apply(x2, self.gradmultiply)
        dropnet = self.get_ratio()
        x = residual + dropnet[0] * x1 + dropnet[1] * x2
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = args.decoder_embed_dim
        self.dropout_module = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim, args, add_bias_kv=add_bias_kv, add_zero_attn=add_zero_attn,
        )

        self.plm_attn = self.build_decoder_plm_attention(self.embed_dim, args)

        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.decoder_normalize_before

        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.fc1 = self.build_fc1(
            self.embed_dim,
            args.decoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            args.decoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True

        self.onnx_trace = False
        self.dropnet = getattr(args, "dropnet", 0.25)
        self.gradmultiply = getattr(args, "gradmultiply", 1.)
        self.plm_as_encoder = getattr(args, "plm_as_encoder", False)

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_decoder_plm_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            kdim=args.plm_encoder_embed_dim,
            vdim=args.plm_encoder_embed_dim,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + x

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def get_ratio(self):
        if self.plm_as_encoder:
            return [0, 1]
        if self.dropnet > 0 and self.training:
            frand = float(uniform(0, 1))
            if frand < self.dropnet:
                return [1, 0]
            elif frand > 1 - self.dropnet:
                return [0, 1]
            else:
                return [0.5, 0.5]
        else:
            return [0.5, 0.5]

    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        plm_out: Optional[torch.Tensor] = None,
        plm_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
        )
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x1, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
            )
            x1 = self.dropout_module(x1)
            x2, _ = self.plm_attn(
                query=x,
                key=plm_out,
                value=plm_out,
                key_padding_mask=plm_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=False,
                need_head_weights=False,
            )
            x2 = GradMultiply.apply(x2, self.gradmultiply)
            x2 = self.dropout_module(x2)
            dropnet = self.get_ratio()
            x = residual + dropnet[0] * x1 + dropnet[1] * x2
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)

        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None


@register_model_architecture("retrosys", "retrosys")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(args, "share_decoder_input_output_embed", False)
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(args, "no_token_positional_embeddings", False)
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(args, "decoder_output_dim", args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.checkpoint_activations = getattr(args, "checkpoint_activations", False)
    args.offload_activations = getattr(args, "offload_activations", False)
    if args.offload_activations:
        args.checkpoint_activations = True
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)
    args.decoder_layers_to_keep = getattr(args, "decoder_layers_to_keep", None)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)


@register_model_architecture("retrosys", "transformer_iwslt_de_en_retrosys")
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    base_architecture(args)
