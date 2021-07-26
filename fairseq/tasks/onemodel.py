from fairseq.data.molecule.graphseq_pair_dataset import MaskGraphDataset
from fairseq.data.lru_cache_dataset import LRUCacheDataset
from fairseq.data.mask_tokens_dataset import MaskTokensDataset
import logging
import os
import numpy as np
from numpy.lib.function_base import append
from fairseq.data import (IdDataset, NestedDictionaryDataset, OffsetTokensDataset,
                          StripTokenDataset, NumSamplesDataset, NumelDataset, data_utils,
                          LeftPadDataset, BaseWrapperDataset, AppendTokenDataset, numel_dataset)
from fairseq.data.shorten_dataset import TruncateDataset, maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from dataclasses import dataclass, field
from typing import Optional, List, Sequence
from omegaconf import II
from fairseq.data.indexed_dataset import (MMapIndexedDataset, get_available_dataset_impl,
                                          make_dataset, infer_dataset_impl)
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from fairseq.data.molecule.molecule import Tensor2Data, MaskedPyGDataset
from torch_geometric.data import Data, Batch
from fairseq import utils
from fairseq.tasks.doublemodel import NoiseOrderedDataset, StripTokenDatasetSizes

logger = logging.getLogger(__name__)


@dataclass
class OneModelConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help":
            "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    left_pad_source: bool = field(default=True, metadata={"help": "pad the source on the left"})
    max_positions: int = II("model.max_positions")
    truncate_source: bool = field(default=False,
                                  metadata={"help": "truncate source to max-source-positions"})
    train_subset: str = II("dataset.train_subset")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II("dataset.dataset_impl")
    required_seq_len_multiple: int = II("dataset.required_seq_len_multiple")
    seed: int = II("common.seed")
    seq_mask_prob: float = field(default=0.15)
    graph_mask_prob: float = field(default=0.15)
    order_noise: int = field(default=5)
    datatype: str = II("model.datatype")


@register_task("onemodel", dataclass=OneModelConfig)
class OneModel(FairseqTask):

    cfg: OneModelConfig

    def __init__(self, cfg: OneModelConfig, src_dict):
        super().__init__(cfg)
        self.src_dict = src_dict
        self.seed = cfg.seed
        self.mask_idx = src_dict.add_symbol("[MASK]")
        self.seq_mask_prob = cfg.seq_mask_prob
        self.graph_mask_prob = cfg.graph_mask_prob
        self.order_noise = cfg.order_noise
        self.datatype = cfg.datatype
        self._max_positions = cfg.max_positions

    @classmethod
    def setup_task(cls, cfg: OneModelConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) == 1

        path = paths[0]
        src_dict = cls.load_dictionary(os.path.join(path, "dict.txt"))
        logger.info("Dictionary {}: {} types.".format(os.path.join(paths[0], "dict.txt"),
                                                      len(src_dict)))
        return cls(cfg, src_dict)

    def load_dataset(self, split: str, combine=False, **kwargs):
        prefix = os.path.join(self.cfg.data, split)
        if self.datatype == 'g':
            dataset = self.load_graph_dataset(prefix=prefix)
        elif self.datatype == 't':
            dataset = self.load_seq_dataset(prefix=prefix)
        else:
            raise NotImplementedError()
        logger.info("Loaded {} with #samples: {}".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def load_graph_dataset(self, prefix: str, **kwargs):
        if not MolMMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("Graph data {} not found.".format(prefix))
        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        dataset = make_graph_dataset(prefix, impl=dataset_impl)
        assert dataset is not None
        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(dataset))

        dataset = Tensor2Data(dataset)
        src_dataset, tgt_dataset = MaskedPyGDataset.apply_mask(dataset=dataset,
                                                               seed=self.cfg.seed,
                                                               mask_prob=self.graph_mask_prob)
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "graph": src_dataset,
            },
            "target": tgt_dataset,
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True)
        }
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(nested_dataset,
                                      sort_order=[shuffle, src_dataset.sizes],
                                      seed=self.cfg.seed,
                                      order_noise=self.order_noise)
        return dataset

    def load_seq_dataset(self, prefix: str, **kwargs):
        if not MMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("SMILES data {} not found.".format(prefix))
        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        dataset = make_dataset(prefix, impl=dataset_impl)
        assert dataset is not None
        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(dataset))
        if self.cfg.truncate_source:
            dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDatasetSizes(dataset, self.source_dictionary.eos()),
                    self._max_positions - 1,
                ), self.source_dictionary.eos())

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            mask_prob=self.seq_mask_prob)
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": LeftPadDataset(src_dataset, pad_idx=self.source_dictionary.pad()),
                "src_lengths": NumelDataset(src_dataset)
            },
            "target": LeftPadDataset(tgt_dataset, pad_idx=self.source_dictionary.pad()),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True)
        }
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(nested_dataset,
                                      sort_order=[shuffle, src_dataset.sizes],
                                      seed=self.cfg.seed,
                                      order_noise=self.order_noise)
        return dataset

    @property
    def source_dictionary(self):
        return self.src_dict

    @property
    def target_dictionary(self):
        return self.src_dict
