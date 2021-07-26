# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils, BaseWrapperDataset
from torch_geometric.data import Data, Batch
from functools import lru_cache
from molecule.features import get_mask_atom_typeid, get_mask_edge_typeid


logger = logging.getLogger(__name__)


def collate(
        samples,
        pad_idx,
        eos_idx,
        left_pad_seq=True,
        pad_to_multiple=1
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None,
        padding_idx=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx if padding_idx is None else padding_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple
        )

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "seq",
        left_pad=left_pad_seq,
        pad_to_length=None
    )
    masked_tokens_label = merge(
        "seq_masked_label",
        left_pad=left_pad_seq,
        pad_to_length=None
    )
    masked_pos = merge(
        "seq_masked_pos",
        left_pad=left_pad_seq,
        pad_to_length=None,
        padding_idx=False
    )
    src_lengths = torch.LongTensor(
        [s["seq"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    masked_pos = masked_pos.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)
    masked_tokens_label = masked_tokens_label.index_select(0, sort_order)
    ntokens = src_lengths.sum().item()
    graph_list = [samples[i]['graph'] for i in sort_order.tolist()]
    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "seq_masked_pos": masked_pos,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths
        },
        "masked_tokens_label": masked_tokens_label,
        "graph": Batch.from_data_list(graph_list)
    }
    return batch


class GraphSeqPairDataset(FairseqDataset):

    def __init__(self,
                 seq,
                 seq_sizes,
                 seq_dict,
                 graph,
                 graph_sizes,
                 left_pad_seq=True,
                 shuffle=True,
                 remove_eos_from_seq=False,
                 constraints=None,
                 append_bos=False,
                 eos=None,
                 num_buckets=0,
                 pad_to_multiple=1,
                 mask_idx=None,
                 seed=0,
                 mask_prob=0.15,
                 graph_mask_prob=0.15,
                 order_noise=3):
        assert len(seq) == len(graph)
        self.seq = MaskTokensDataSet(
            dataset=seq,
            vocab=seq_dict,
            pad_idx=seq_dict.pad(),
            mask_idx=mask_idx,
            mask_prob=mask_prob
        )
        self.graph = MaskGraphDataset(
            graph,
            mask_prob=graph_mask_prob
        )
        self.seq_sizes = seq_sizes
        self.graph_sizes = graph_sizes
        self.seq_dict = seq_dict

        self.left_pad_seq = left_pad_seq
        self.shuffle = shuffle
        self.remove_eos_from_seq = remove_eos_from_seq
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else seq_dict.eos()
        if num_buckets > 0:
            raise NotImplementedError()
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple
        self.order_noise = order_noise
        if self.order_noise > 0:
            self._can_reuse_epoch_itr_across_epochs = False
        else:
            assert self.order_noise == 0
            self._can_reuse_epoch_itr_across_epochs = True
        self.seed = seed
        self._epoch = 0

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        graph_item = self.graph[index]
        masked_tokens, masked_labels, masked_pos = self.seq[index]

        if self.append_bos:
            bos = self.seq_dict.bos()
            if masked_tokens[0] != bos:
                masked_tokens = torch.cat([torch.LongTensor([bos]), masked_tokens])
                masked_labels = torch.cat([torch.LongTensor([bos]), masked_labels])

        if self.remove_eos_from_seq:
            if masked_tokens[-1] == self.eos:
                masked_tokens = masked_tokens[:-1]
                masked_labels = masked_labels[:-1]
        graph_item = Data(x=graph_item['node_attr'],
                          edge_index=graph_item['edge_index'].T,
                          edge_attr=graph_item['edge_attr'],
                          x_label=graph_item['node_attr_label'],
                          masked_pos=graph_item['node_masked_pos'])
        example = {
            "id": index,
            "seq": masked_tokens,
            "seq_masked_label": masked_labels,
            "seq_masked_pos": masked_pos,
            "graph": graph_item
        }
        return example

    def __len__(self):
        return len(self.seq)

    def collater(self, samples):
        res = collate(
            samples,
            pad_idx=self.seq_dict.pad(),
            eos_idx=self.eos,
            left_pad_seq=self.left_pad_seq,
            pad_to_multiple=self.pad_to_multiple
        )
        return res

    def num_tokens(self, index):
        return max(
            self.seq_sizes[index],
            self.graph_sizes[index]
        )

    def num_tokens_vec(self, indices):
        sizes = self.seq_sizes[indices]
        return sizes

    def size(self, index):
        return (
            self.seq_sizes[index],
            self.graph_sizes[index]
        )

    def ordered_indices(self):
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)

        if self.buckets is None:
            if self._can_reuse_epoch_itr_across_epochs:
                if self.graph_sizes is not None:
                    indices = indices[np.argsort(self.graph_sizes[indices], kind='mergesort')]
                return indices[np.argsort(self.seq_sizes[indices], kind='mergesort')]
            else:
                with data_utils.numpy_seed(self.seed + self._epoch):
                    if self.graph_sizes is not None:
                        indices = indices[np.argsort(self.graph_sizes[indices] + np.random.randint(
                            low=-self.order_noise,
                            high=self.order_noise,
                            size=self.graph_sizes.size
                        ), kind='mergesort')]
                    return indices[np.argsort(self.seq_sizes[indices] + np.random.randint(
                        low=-self.order_noise,
                        high=self.order_noise,
                        size=self.seq_sizes.size
                    ), kind='mergesort')]
        else:
            raise NotImplementedError()

    @property
    def supports_prefetch(self):
        return getattr(self.seq, "supports_prefetch", False) and \
            getattr(self.graph, "supports_prefetch", False)

    def prefetch(self, indices):
        self.seq.prefetch(indices)
        self.graph.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        return data_utils.filter_paired_dataset_indices_by_size(
            self.seq_sizes, None, indices, max_sizes
        )

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return self._can_reuse_epoch_itr_across_epochs

    def set_epoch(self, epoch):
        self._epoch = epoch
        super().set_epoch(epoch)


class MaskTokensDataSet(BaseWrapperDataset):

    def __init__(self,
                 dataset,
                 vocab,
                 pad_idx,
                 mask_idx,
                 mask_prob=0.15,
                 leave_unmasked_prob=0.1,
                 random_token_prob=0.1,
                 ):
        super().__init__(dataset)
        self.vocab = vocab
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.mask_prob = mask_prob
        self.leave_unmasked_prob = leave_unmasked_prob
        self.random_token_prob = random_token_prob

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        item = self.dataset[index]
        sz = len(item)

        assert self.mask_idx not in item

        mask = np.full(sz, False)
        num_mask = int(
            self.mask_prob * sz + np.random.rand()
        )
        mask_idc = np.random.choice(sz, num_mask, replace=False)
        # we do not want to mask eos
        mask_idc = mask_idc[mask_idc < len(mask) - 1]
        mask[mask_idc] = True
        masked_labels = np.full(len(mask), self.pad_idx)
        masked_labels[mask] = item[torch.from_numpy(mask.astype(np.uint8)) == 1]
        masked_labels = torch.from_numpy(masked_labels)
        masked_pos = mask

        rand_or_unmask_prob = self.random_token_prob + self.leave_unmasked_prob
        if rand_or_unmask_prob > 0.0:
            rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
            if self.random_token_prob == 0:
                unmask = rand_or_unmask
                rand_mask = None
            elif self.leave_unmasked_prob == 0:
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

        masked_tokens = np.copy(item)
        masked_tokens[mask] = self.mask_idx
        if rand_mask is not None:
            num_rand = rand_mask.sum()
            if num_rand > 0:
                masked_tokens[rand_mask] = np.random.choice(
                    len(self.vocab),
                    num_rand,
                )
        return torch.from_numpy(masked_tokens), masked_labels, torch.from_numpy(masked_pos)


class MaskGraphDataset(BaseWrapperDataset):

    def __init__(self,
                 dataset,
                 mask_prob=0.15):
        super().__init__(dataset)
        self.node_pad_idx = get_mask_atom_typeid()
        self.edge_pad_idx = get_mask_edge_typeid()
        self.node_mask_idx = self.node_pad_idx
        self.edge_mask_idx = self.edge_pad_idx
        self.mask_prob = mask_prob

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        graph_item = self.dataset[index]
        x = graph_item['node_attr']
        sz = x.size(0)
        assert self.node_mask_idx not in x[:, 0]
        mask = np.full(sz, False)
        num_mask = int(self.mask_prob * sz + np.random.rand())
        mask_idc = np.random.choice(sz, num_mask, replace=False)
        mask_idc = mask_idc[mask_idc < len(mask)]
        mask[mask_idc] = True
        masked_labels = x.detach().clone()
        masked_labels[:, 0] = self.node_pad_idx
        masked_labels[mask] = x[torch.from_numpy(mask.astype(np.uint8)) == 1]
        masked_x = x.detach().clone()
        masked_x[mask] = 0
        masked_x[mask, 0] = self.node_mask_idx
        graph_item['node_attr'] = masked_x
        graph_item['node_attr_label'] = masked_labels
        graph_item['node_masked_pos'] = torch.from_numpy(mask)  

        return graph_item





