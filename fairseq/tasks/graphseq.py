from dataclasses import dataclass, field
import itertools
import json
import logging
import os
import numpy as np
from typing import Optional, List
from omegaconf import II
from fairseq import metrics, utils
from fairseq.data.indexed_dataset import (get_available_dataset_impl,
                                          make_dataset,
                                          infer_dataset_impl)
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.graphseq_pair_dataset import GraphSeqPairDataset
from fairseq.dataclass import ChoiceEnum, FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from fairseq.data import (
    AppendTokenDataset,
    TruncateDataset,
    StripTokenDataset
)
import torch


logger = logging.getLogger(__name__)

def load_graphseq_dataset(
        data_path,
        split,
        seq_dict,
        combine,
        dataset_impl,
        left_pad_seq,
        max_source_positions,
        prepend_bos=False,
        truncate_source=False,
        num_buckets=0,
        shuffle=True,
        pad_to_multiple=1,
        prepend_bos_src=None,
        mask_idx=None,
        mask_prob=0.15,
        seed=0,
        graph_mask_prob=0.15,
        order_noise=3
):

    seq_datasets = []
    graph_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")
        prefix = os.path.join(data_path, split_k)

        if not MolMMapIndexedDataset.exists(prefix):
            if k > 0:
                break
            else:
                raise FileNotFoundError("Graph Data {} not found.".format(prefix))
        if dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)

        seq_dataset = make_dataset(prefix, impl=dataset_impl)
        if truncate_source:
            seq_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(seq_dataset, seq_dict.eos()),
                    max_source_positions - 1,
                ),
                seq_dict.eos()
            )
        seq_datasets.append(seq_dataset)

        graph_dataset = make_graph_dataset(prefix, impl=dataset_impl)
        graph_datasets.append(graph_dataset)
        logger.info(
            "{} {} {} examples".format(data_path, split_k, len(seq_dataset))
        )
        if not combine:
            break

    assert len(seq_datasets) == len(graph_datasets)
    if len(seq_datasets) == 1:
        seq_dataset = seq_datasets[0]
        graph_dataset = graph_datasets[0]
    else:
        raise NotImplementedError()
    if prepend_bos:
        raise NotImplementedError()
    elif prepend_bos_src is not None:
        raise NotImplementedError()

    return GraphSeqPairDataset(
        seq_dataset,
        seq_dataset.sizes,
        seq_dict,
        graph_dataset,
        graph_dataset.sizes,
        left_pad_seq=left_pad_seq,
        eos=None,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
        num_buckets=0,
        mask_idx=mask_idx,
        mask_prob=mask_prob,
        seed=seed,
        graph_mask_prob=graph_mask_prob,
        order_noise=order_noise
    )


@dataclass
class GraphSeqConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
                    "in round-robin manner; however, valid and test data are always in the first directory "
                    "to avoid the need for repeating them in all directories"
        },
    )
    load_alignments: bool = field(
        default=False, metadata={"help": "load the binarized alignments"}
    )
    left_pad_source: bool = field(
        default=True, metadata={"help": "pad the source on the left"}
    )
    max_source_positions: int = II("model.max_positions")
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
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")
    seed: int = II("common.seed")
    seq_mask_prob: float = field(default=0.15)
    graph_mask_prob: float = field(default=0.15)
    order_noise: int = field(default=5)


@register_task("graphseq", dataclass=GraphSeqConfig)
class GraphSeqTask(FairseqTask):

    cfg: GraphSeqConfig

    def __init__(self, cfg: GraphSeqConfig, src_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.seed = cfg.seed
        self.mask_idx = src_dict.add_symbol("[MASK]")
        self.seq_mask_prob = cfg.seq_mask_prob
        self.graph_mask_prob = cfg.graph_mask_prob
        self.order_noise = cfg.order_noise

    @classmethod
    def setup_task(cls, cfg: GraphSeqConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) > 0

        src_dict = cls.load_dictionary(os.path.join(paths[0], "dict.txt"))
        logger.info("Dictionary {}: {} types.".format(os.path.join(paths[0], "dict.txt"), len(src_dict)))
        return cls(cfg, src_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        if split != self.cfg.train_subset:
            paths = paths[:1]

        data_path = paths[(epoch - 1) % len(paths)]

        self.datasets[split] = load_graphseq_dataset(
            data_path,
            split,
            self.src_dict,
            combine=combine,
            dataset_impl=self.cfg.dataset_impl,
            left_pad_seq=self.cfg.left_pad_source,
            max_source_positions=self.cfg.max_source_positions,
            truncate_source=self.cfg.truncate_source,
            num_buckets=self.cfg.num_batch_buckets,
            shuffle=(split != "test"),
            pad_to_multiple=self.cfg.required_seq_len_multiple,
            mask_idx=self.mask_idx,
            seed=self.seed,
            mask_prob=self.seq_mask_prob,
            graph_mask_prob=self.graph_mask_prob,
            order_noise=self.order_noise
        )

    def build_dataset_for_inference(
        self, src_tokens: List[torch.Tensor], src_lengths: List[int], **kwargs
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError()

    def build_model(self, cfg):
        model = super().build_model(cfg)
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.src_dict
