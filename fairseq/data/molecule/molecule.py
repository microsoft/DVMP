# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from functools import lru_cache
from fairseq.data.lru_cache_dataset import LRUCacheDataset
from fairseq.data import BaseWrapperDataset, data_utils
from torch_geometric.data import Data
import torch
from torch_geometric.data import Batch
from molecule.features import (get_mask_atom_typeid, get_atom_feature_dims, get_mask_atom_feature)
import numpy as np


class Tensor2Data(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        graph_item = Data(x=item["node_attr"], edge_index=item["edge_index"].T, edge_attr=item['edge_attr'])
        return graph_item

    def collater(self, samples):
        return mol_collater(samples)


def mol_collater(samples):
    if isinstance(samples[0], Data):
        return Batch.from_data_list(samples)
    elif isinstance(samples[0], torch.Tensor):
        # for target labels
        return torch.cat(samples, dim=0)
    else:
        raise NotImplementedError()


class MaskedPyGDataset(BaseWrapperDataset):
    @classmethod
    def apply_mask(cls, dataset: torch.utils.data.Dataset, *args, **kwargs):
        dataset = LRUCacheDataset(dataset)
        return (LRUCacheDataset(cls(dataset, *args, **kwargs,
                                    return_masked_tokens=False)), LRUCacheDataset(cls(dataset, *args, **kwargs, return_masked_tokens=True)))

    def __init__(
        self,
        dataset: torch.utils.data.Dataset,
        vocab=None,
        pad_idx: int = get_mask_atom_typeid(),
        mask_idx: int = get_mask_atom_typeid(),
        return_masked_tokens: bool = False,
        seed: int = 1,
        mask_prob: float = 0.15,
        leave_unmasked_prob: float = 0.1,
        random_token_prob: float = 0.1,
        freq_weighted_replacement: bool = False,
    ):
        assert 0.0 < mask_prob < 1.0
        assert 0.0 <= random_token_prob <= 1.0
        assert 0.0 <= leave_unmasked_prob <= 1.0
        assert random_token_prob + leave_unmasked_prob <= 1.0

        self.dataset = dataset
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.return_masked_tokens = return_masked_tokens
        self.seed = seed
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob
        self.mask_feature = get_mask_atom_feature()

        if random_token_prob > 0.0:
            if freq_weighted_replacement:
                raise NotImplementedError()
            else:
                weights = np.ones(get_atom_feature_dims()[0])
            weights[get_mask_atom_typeid():] = 0
            self.weights = weights / weights.sum()
        self.epoch = 0

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        self.epoch = epoch

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

    def __getitem__(self, index):
        return self.__getitem_cached__(index)

    @lru_cache(maxsize=8)
    def __getitem_cached__(self, index: int):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            graph_item = self.dataset[index]
            item = graph_item.x
            sz = item.size(0)

            assert self.mask_idx not in item
            assert item.size(-1) == len(self.mask_feature)
            mask = np.full(sz, False)
            num_mask = int(self.mask_prob * sz + np.random.rand())
            mask_idc = np.random.choice(sz, num_mask, replace=False)
            mask_idc = mask_idc[mask_idc < len(mask)]
            try:
                mask[mask_idc] = True
            except Exception as e:
                print("Assigning mask indexes {} to mask {} failed!".format(mask_idc, mask))
                raise e
            if self.return_masked_tokens:
                new_item = np.full(item.size(), self.mask_feature)
                new_item[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
                return torch.from_numpy(new_item[:, 0])

            rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
            if rand_or_unmask_prob > 0.0:
                rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
                if self.random_token_prob == 0.0:
                    unmask = rand_or_unmask
                    rand_mask = None
                elif self.leave_unmasked_prob == 0.0:
                    unmask = None
                    rand_mask = rand_or_unmask
                else:
                    unmask_prob = self.leave_unmasked_prob / rand_or_unmask_prob
                    decision = np.random.rand(sz) < unmask_prob
                    unmask = rand_or_unmask & decision
                    rand_mask = rand_or_unmask & (~decision)
            else:
                unmask = rand_mask = None

            if unmask is not None:
                mask = mask ^ unmask

            new_item = np.copy(item)
            new_item[mask] = self.mask_feature
            if rand_mask is not None:
                num_rand = rand_mask.sum()
                if num_rand > 0:
                    new_index = np.random.choice(
                        len(item),
                        num_rand,
                    )
                    new_item[rand_mask] = item.index_select(0, torch.from_numpy(new_index))

            new_graph = graph_item.clone()
            new_graph.x = torch.from_numpy(new_item)
            return new_graph
