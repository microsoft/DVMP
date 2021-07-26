from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from omegaconf import II

import numpy as np
from fairseq import metrics, utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
    BaseWrapperDataset,
    IdDataset,
    LeftPadDataset,
    NumelDataset,
    NumSamplesDataset,
    NestedDictionaryDataset,
)
from fairseq.data.indexed_dataset import (
    get_available_dataset_impl,
    infer_dataset_impl,
    MMapIndexedDataset,
    make_dataset,
)
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from fairseq.data.molecule.molecule import Tensor2Data
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.translation import load_langpair_dataset

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class RetroSysConfig(FairseqDataclass):
    data: Optional[str] = field(default=None)
    source_lang: Optional[str] = field(default=None)
    target_lang: Optional[str] = field(default=None)

    load_alignments: bool = field(default=False, metadata={"help": "load the binarized alignments"})
    left_pad_source: bool = field(default=True, metadata={"help": "pad the source on the left"})
    left_pad_target: bool = field(default=False, metadata={"help": "pad the target on the left"})
    max_source_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024, metadata={"help": "max number of tokens in the target sequence"}
    )
    upsample_primary: int = field(
        default=-1, metadata={"help": "the amount of upsample primary dataset"}
    )
    truncate_source: bool = field(
        default=False, metadata={"help": "truncate source to max-source-positions"}
    )
    num_batch_buckets: int = field(
        default=0,
        metadata={
            "help": "if >0, then bucket source and target lengths into "
            "N buckets and pad accordingly; this is useful on TPUs to minimize the number of compilations"
        },
    )
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II("dataset.dataset_impl")
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")

    # options for reporting BLEU during validation
    eval_bleu: bool = field(default=False, metadata={"help": "evaluation with BLEU scores"})
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={"help": "remove BPE before computing BLEU", "argparse_const": "@@ ",},
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    # options for pre-tranined language model
    datatype: str = field(default="tg")
    plm_max_positions: int = field(default=512)


@register_task("retrosys", dataclass=RetroSysConfig)
class RetroSysTask(FairseqTask):

    cfg: RetroSysConfig

    def __init__(self, cfg: RetroSysConfig, src_dict, tgt_dict, plm_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.plm_dict = plm_dict
        plm_dict.add_symbol("[MASK]")
        self.plm_max_positions = cfg.plm_max_positions

    @classmethod
    def setup_task(cls, cfg: RetroSysConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) == 1

        # find language pair automatically
        if cfg.source_lang is None or cfg.target_lang is None:
            cfg.source_lang, cfg.target_lang = data_utils.infer_language_pair(
                os.path.join(paths[0], "input0")
            )
        if cfg.source_lang is None or cfg.target_lang is None:
            raise Exception("Could not infer language pair, please provide it explicitly")
        src_dict = cls.load_dictionary(
            os.path.join(paths[0], "input0", "dict.{}.txt".format(cfg.source_lang))
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(paths[0], "input0", "dict.{}.txt".format(cfg.target_lang))
        )
        assert src_dict.pad() == tgt_dict.pad()
        assert src_dict.eos() == tgt_dict.eos()
        assert src_dict.unk() == tgt_dict.unk()
        plm_dict = cls.load_dictionary(os.path.join(paths[0], "input1", "dict.txt"))

        logger.info("[{}] dictionary: {} types".format(cfg.source_lang, len(src_dict)))
        logger.info("[{}] dictionary: {} types".format(cfg.target_lang, len(tgt_dict)))
        logger.info("PLM dictionary: {} types".format(len(plm_dict)))

        return cls(cfg, src_dict, tgt_dict, plm_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) == 1
        data_path = paths[0]

        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        pair_dataset = load_langpair_dataset(
            os.path.join(data_path, "input0"),
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=self.cfg.max_source_positions,
            max_target_positions=self.cfg.max_target_positions,
            load_alignments=self.cfg.load_alignments,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
        )
        if self.cfg.datatype == "tg" or self.datattype == "gt":
            plm_dataset = self.load_tg_dataset(os.path.join(data_path, "input1", split))
        else:
            raise NotImplementedError()
        self.datasets[split] = PLMDataset(pair_dataset, plm_dataset)

    def load_tg_dataset(self, prefix: str, **kwargs):
        if not MMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("SMILES data {} not found.".format(prefix))
        if not MolMMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("PyG data {} not found.".format(prefix))
        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        src_dataset = make_dataset(prefix, impl=dataset_impl)
        assert src_dataset is not None

        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDataset(src_dataset, self.plm_dict.eos()), self.plm_max_positions - 1
            ),
            self.plm_dict.eos(),
        )

        src_dataset_graph = make_graph_dataset(prefix, impl=dataset_impl)
        assert src_dataset_graph is not None

        src_dataset_graph = Tensor2Data(src_dataset_graph)

        dataset = {
            "id": IdDataset(),
            "net_input0": {
                "src_tokens": LeftPadDataset(src_dataset, pad_idx=self.plm_dict.pad()),
                "src_length": NumelDataset(src_dataset),
            },
            "net_input1": {"graph": src_dataset_graph,},
        }
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        return nested_dataset

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        raise NotImplementedError()

    def build_model(self, cfg):
        return super().build_model(cfg)

    def max_positions(self):
        return (self.cfg.max_source_positions, self.cfg.max_target_positions)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.tgt_dict


class PLMDataset(BaseWrapperDataset):
    def __init__(self, pair_datatset, plm_dataset):
        super().__init__(pair_datatset)
        self.plm_dataset = plm_dataset

    def __getitem__(self, index):
        example = super().__getitem__(index)
        plm_example = self.plm_dataset[index]
        return (example, plm_example)

    def collater(self, samples):
        samples_0 = [sample[0] for sample in samples]
        samples_1 = [sample[1] for sample in samples]
        samples = super().collater(samples_0)
        if len(samples) == 0:
            return samples
        samples["net_input"]["plm_input"] = self.plm_dataset.collater(samples_1)
        return samples
