# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
from torch_geometric.data import Data, Batch
from fairseq import utils
from fairseq.data.molecule.molecule import Tensor2Data, MaskedPyGDataset

logger = logging.getLogger(__name__)


@dataclass
class DoubleModelConfig(FairseqDataclass):
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


@register_task('dmp', dataclass=DoubleModelConfig)
class DoubleModel(FairseqTask):

    cfg: DoubleModelConfig

    def __init__(self, cfg: DoubleModelConfig, src_dict):
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
    def setup_task(cls, cfg: DoubleModelConfig, **kwargs):
        paths = utils.split_paths(cfg.data)
        assert len(paths) == 1

        path = paths[0]
        src_dict = cls.load_dictionary(os.path.join(path, "dict.txt"))
        logger.info("Dictionary {}: {} types.".format(os.path.join(paths[0], "dict.txt"),
                                                      len(src_dict)))
        return cls(cfg, src_dict)

    def load_dataset(self, split: str, combine=False, **kwargs):
        prefix = os.path.join(self.cfg.data, split)
        if self.datatype == 'tt':
            dataset = self.load_tt_dataset(prefix)
        elif self.datatype == "tg" or self.datatype == "gt":
            dataset = self.load_tg_dataset(prefix)
        elif self.datatype == "gg":
            dataset = self.load_gg_dataset(prefix)
        else:
            raise NotImplementedError()

        logger.info("Loaded {} with #samples: {}".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def load_tt_dataset(self, prefix: str, **kwargs):
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

        dataset = LRUCacheDataset(dataset)
        src_dataset0, tgt_dataset0 = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            mask_prob=self.seq_mask_prob,
        )
        src_dataset1, tgt_dataset1 = MaskTokensDataset.apply_mask(
            dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed + 1,
            mask_prob=self.seq_mask_prob)

        dataset = {
            "id": IdDataset(),
            "net_input0": {
                "src_tokens": LeftPadDataset(src_dataset0, pad_idx=self.source_dictionary.pad()),
                "src_lengths": NumelDataset(src_dataset0)
            },
            "target0": LeftPadDataset(tgt_dataset0, pad_idx=self.source_dictionary.pad()),
            "net_input1": {
                "src_tokens": LeftPadDataset(src_dataset1, pad_idx=self.source_dictionary.pad()),
                "src_lengths": NumelDataset(src_dataset1)
            },
            "target1": LeftPadDataset(tgt_dataset1, pad_idx=self.source_dictionary.pad()),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset0, reduce=True),
        }
        nestd_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset0.sizes])
        dataset = NoiseOrderedDataset(nestd_dataset,
                                      sort_order=[shuffle, src_dataset0.sizes],
                                      seed=self.cfg.seed,
                                      order_noise=self.order_noise)
        return dataset

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

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        if self.cfg.truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDatasetSizes(src_dataset, self.source_dictionary.eos()),
                    self._max_positions - 1,
                ), self.source_dictionary.eos())

        src_dataset_graph = make_graph_dataset(prefix, impl=dataset_impl)
        assert src_dataset_graph is not None

        src_dataset_graph = Tensor2Data(src_dataset_graph)

        src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
            src_dataset,
            self.source_dictionary,
            pad_idx=self.source_dictionary.pad(),
            mask_idx=self.mask_idx,
            seed=self.cfg.seed,
            mask_prob=self.seq_mask_prob)
        src_dataset_graph, tgt_dataset_graph = MaskedPyGDataset.apply_mask(
            dataset=src_dataset_graph, seed=self.cfg.seed, mask_prob=self.graph_mask_prob)

        dataset = {
            "id": IdDataset(),
            "net_input0": {
                "src_tokens": LeftPadDataset(src_dataset, pad_idx=self.source_dictionary.pad()),
                "src_length": NumelDataset(src_dataset)
            },
            "target0": LeftPadDataset(tgt_dataset, pad_idx=self.source_dictionary.pad()),
            "net_input1": {
                "graph": src_dataset_graph
            },
            "target1": tgt_dataset_graph,
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }
        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(nested_dataset,
                                      sort_order=[shuffle, src_dataset.sizes],
                                      seed=self.cfg.seed,
                                      order_noise=self.order_noise)
        return dataset

    def load_gg_dataset(self, prefix: str, **kwargs):
        if not MolMMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("PyG data {} not found.".format(prefix))

        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        src_dataset = make_graph_dataset(prefix, impl=dataset_impl)
        assert src_dataset is not None

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        src_dataset = Tensor2Data(src_dataset)
        src_dataset = LRUCacheDataset(src_dataset)
        src_dataset0, tgt_dataset0 = MaskedPyGDataset.apply_mask(dataset=src_dataset,
                                                                 seed=self.cfg.seed,
                                                                 mask_prob=self.graph_mask_prob)
        src_dataset1, tgt_dataset1 = MaskedPyGDataset.apply_mask(dataset=src_dataset,
                                                                 seed=self.cfg.seed + 1,
                                                                 mask_prob=self.graph_mask_prob)
        dataset = {
            "id": IdDataset(),
            "net_input0": {
                "graph": src_dataset0
            },
            "target0": tgt_dataset0,
            "net_input1": {
                "graph": src_dataset1
            },
            "target1": tgt_dataset1,
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


class StripTokenDatasetSizes(StripTokenDataset):
    def __init__(self, dataset, id_to_strip):
        super().__init__(dataset, id_to_strip)
        self._sizes = np.array(dataset.sizes) - 1

    @property
    def sizes(self):
        return self._sizes


class NoiseOrderedDataset(BaseWrapperDataset):
    def __init__(self, dataset, sort_order, seed, order_noise):
        super().__init__(dataset)
        if not isinstance(sort_order, (list, tuple)):
            sort_order = [sort_order]
        self.sort_order = sort_order
        self.seed = seed
        self.order_noise = order_noise

        assert all(len(so) == len(dataset) for so in sort_order)
        self._epoch = 0

    def ordered_indices(self):
        sort_order = []
        with data_utils.numpy_seed(self.seed + self._epoch):
            for so in self.sort_order:
                sort_order.append(
                    so +
                    np.random.randint(low=-self.order_noise, high=self.order_noise, size=so.shape))
            return np.lexsort(sort_order)

    def set_epoch(self, epoch):
        self._epoch = epoch
        super().set_epoch(epoch)

    def num_tokens_vec(self, indices):
        if isinstance(self.sizes, list):
            return self.sizes[0][indices]
        elif isinstance(self.sizes, np.ndarray):
            return self.sizes[indices]
        else:
            raise NotImplementedError()

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def batch_by_size(self, indices, max_tokens, max_sentences, required_batch_size_multiple):
        batch_sampler = super().batch_by_size(
            indices,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple)
        return [x for x in batch_sampler if len(x) > 1]
