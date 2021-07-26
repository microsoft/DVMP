# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from fairseq.data.lru_cache_dataset import LRUCacheDataset
from fairseq.data.append_token_dataset import AppendTokenDataset
from functools import lru_cache, reduce
import logging
import os
import numpy as np
from numpy.core.fromnumeric import sort
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    NumSamplesDataset,
    NumelDataset,
    data_utils,
    LeftPadDataset,
    BaseWrapperDataset,
    RawLabelDataset,
)
from fairseq.data.shorten_dataset import TruncateDataset, maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from dataclasses import dataclass, field
from typing import Optional, List, Any
from omegaconf import II
from fairseq.data.indexed_dataset import (
    MMapIndexedDataset,
    get_available_dataset_impl,
    make_dataset,
    infer_dataset_impl,
)
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from torch_geometric.data import Data, Batch
from fairseq.data.molecule.molecule import Tensor2Data
from fairseq.tasks.doublemodel import NoiseOrderedDataset, StripTokenDatasetSizes

logger = logging.getLogger(__name__)


@dataclass
class GrapgSenPredictionConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
            "in round-robin manner; however, valid and test data are always in the first directory "
            "to avoid the need for repeating them in all directories"
        },
    )
    num_classes: int = field(default=2)
    regression_target: bool = field(default=False)
    scaler_label: bool = field(default=False)
    no_shuffle: bool = field(default=False)
    shorten_method: ChoiceEnum(["none", "truncate", "random_crop"]) = field(default="truncate")
    shorten_data_split_list: str = field(default="")
    max_positions: int = II("model.max_positions")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II("dataset.dataset_impl")
    seed: int = II("common.seed")
    order_noise: int = field(default=5)
    datatype: str = II("model.datatype")


@register_task("graph_sp", dataclass=GrapgSenPredictionConfig)
class GraphSenPrediction(FairseqTask):

    cfg: GrapgSenPredictionConfig

    def __init__(self, cfg: GrapgSenPredictionConfig, data_dictionary, label_dictionary):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self.dictionary.add_symbol("[MASK]")
        self.label_dictionary = label_dictionary
        self._max_positions = cfg.max_positions
        self.seed = cfg.seed
        self.order_noise = cfg.order_noise
        self.datatype = cfg.datatype
        if self.cfg.regression_target and self.cfg.scaler_label:
            self.prepare_scaler()
        else:
            self.label_scaler = None

    def prepare_scaler(self):
        label_path = "{}.label".format(self.get_path("label", "train"))
        assert os.path.exists(label_path)

        def parse_regression_target(i, line):
            values = line.split()
            assert (
                len(values) == self.cfg.num_classes
            ), f'expected num_classes={self.cfg.num_classes} regression target values on line {i}, found: "{line}"'
            return [float(x) for x in values]

        with open(label_path) as h:
            x = [parse_regression_target(i, line.strip()) for i, line in enumerate(h.readlines())]
        self.label_scaler = StandardScaler(x)

    def inverse_transform(self, x):
        if self.label_scaler is None:
            return x 
        else:
            return self.label_scaler.inverse_transform(x)
    
    def transform_label(self, x):
        if self.label_scaler is None:
            return x
        else:
            return self.label_scaler.transform(x)

    @classmethod
    def setup_task(cls, cfg: GrapgSenPredictionConfig, **kwargs):
        assert cfg.num_classes > 0
        data_dict = cls.load_dictionary(os.path.join(cfg.data, "input0", "dict.txt"))
        logger.info(
            "[input] Dictionary {}: {} types.".format(
                os.path.join(cfg.data, "input0",), len(data_dict)
            )
        )
        if not cfg.regression_target:
            label_dict = cls.load_dictionary(os.path.join(cfg.data, "label", "dict.txt"))
            logger.info(
                "[label] Dictionary {}: {} types.".format(
                    os.path.join(cfg.data, "label"), len(label_dict)
                )
            )
        else:
            label_dict = data_dict
        return cls(cfg, data_dict, label_dict)

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        if self.datatype == "g":
            dataset = self.load_dataset_g(split)
        elif self.datatype == "t":
            dataset = self.load_dataset_tt(split)
        elif self.datatype == "tt":
            dataset = self.load_dataset_tt(split)
        elif self.datatype in ["tg", "gt"]:
            dataset = self.load_dataset_tg(split)
        elif self.datatype == "gg":
            dataset = self.load_dataset_gg(split)
        else:
            raise NotImplementedError()
        logger.info("Loaded {} with #samples: {}.".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def get_path(self, key, split):
        return os.path.join(self.cfg.data, key, split)

    def load_dataset_g(self, split):
        prefix = self.get_path("input0", split)
        if not MolMMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("Graph data {} not found.".format(prefix))

        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        src_dataset = make_graph_dataset(prefix, dataset_impl)
        assert src_dataset is not None
        src_dataset = Tensor2Data(src_dataset)

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        src_dataset = TruncateSizesDataset(src_dataset, self._max_positions)
        dataset = {
            "id": IdDataset(),
            "net_input": {"graph": src_dataset,},
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }
        if not self.cfg.regression_target:
            prefix = self.get_path("label", split)
            label_dataset = make_dataset(prefix, impl=dataset_impl)
            assert label_dataset is not None
            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(label_dataset, id_to_strip=self.label_dictionary.eos()),
                    offset=-self.label_dictionary.nspecial,
                )
            )
        else:
            raise NotImplementedError

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(
            nested_dataset,
            sort_order=[shuffle, src_dataset.sizes],
            seed=self.seed,
            order_noise=self.order_noise,
        )
        return dataset

    def load_dataset_tt(self, split):
        prefix = self.get_path("input0", split)
        if not MMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("Graph data {} not found.".format(prefix))

        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        src_dataset = make_dataset(prefix, impl=dataset_impl)
        assert src_dataset is not None

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDatasetSizes(src_dataset, self.source_dictionary.eos()),
                self._max_positions - 1,
            ),
            self.source_dictionary.eos(),
        )
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": LeftPadDataset(src_dataset, pad_idx=self.source_dictionary.pad()),
                "src_lengths": NumelDataset(src_dataset),
            },
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }

        if not self.cfg.regression_target:
            prefix = self.get_path("label", split)
            label_dataset = make_dataset(prefix, impl=dataset_impl)
            assert label_dataset is not None

            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(label_dataset, id_to_strip=self.label_dictionary.eos()),
                    offset=-self.label_dictionary.nspecial,
                )
            )
        else:
            raise NotImplementedError()

        nesterd_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(
            nesterd_dataset,
            sort_order=[shuffle, src_dataset.sizes],
            seed=self.seed,
            order_noise=self.order_noise,
        )
        return dataset

    def load_dataset_tg(self, split):
        prefix = self.get_path("input0", split)
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

        src_dataset = AppendTokenDataset(
            TruncateDataset(
                StripTokenDatasetSizes(src_dataset, self.source_dictionary.eos()),
                self._max_positions - 1,
            ),
            self.source_dictionary.eos(),
        )

        src_dataset_graph = make_graph_dataset(prefix, impl=dataset_impl)
        assert src_dataset_graph is not None
        src_dataset_graph = Tensor2Data(src_dataset_graph)

        dataset = {
            "id": IdDataset(),
            "net_input0": {
                "src_tokens": LeftPadDataset(src_dataset, pad_idx=self.source_dictionary.pad()),
                "src_lengths": NumelDataset(src_dataset),
            },
            "net_input1": {"graph": src_dataset_graph,},
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }
        if not self.cfg.regression_target:
            prefix = self.get_path("label", split)
            label_dataset = make_dataset(prefix, impl=dataset_impl)
            assert label_dataset is not None

            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(label_dataset, id_to_strip=self.label_dictionary.eos()),
                    offset=-self.label_dictionary.nspecial,
                )
            )
        else:
            label_path = "{}.label".format(self.get_path("label", split))
            assert os.path.exists(label_path)

            def parse_regression_target(i, line):
                values = line.split()
                assert (
                    len(values) == self.cfg.num_classes
                ), f'expected num_classes={self.cfg.num_classes} regression target values on line {i}, found: "{line}"'
                return [self.transform_label(float(x)) for x in values]

            with open(label_path) as h:
                dataset.update(
                    target=RawLabelDataset(
                        [
                            parse_regression_target(i, line.strip())
                            for i, line in enumerate(h.readlines())
                        ]
                    )
                )

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(
            nested_dataset,
            sort_order=[shuffle, src_dataset.sizes],
            seed=self.seed,
            order_noise=self.order_noise,
        )
        return dataset

    def load_dataset_gg(self, split):
        prefix = self.get_path("input0", split)
        if not MolMMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("Graph data {} not found.".format(prefix))

        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        src_dataset = make_graph_dataset(prefix, impl=dataset_impl)
        assert src_dataset is not None
        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(src_dataset))

        src_dataset = Tensor2Data(src_dataset)
        src_dataset = TruncateSizesDataset(src_dataset, self._max_positions)
        dataset = {
            "id": IdDataset(),
            "net_input": {"graph": src_dataset},
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_dataset, reduce=True),
        }
        if not self.cfg.regression_target:
            prefix = self.get_path("label", split)
            label_dataset = make_dataset(prefix, impl=dataset_impl)
            assert label_dataset is not None
            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(label_dataset, id_to_strip=self.label_dictionary.eos()),
                    offset=-self.label_dictionary.nspecial,
                )
            )
        else:
            raise NotImplementedError()

        nested_dataset = NestedDictionaryDataset(dataset, sizes=[src_dataset.sizes])
        dataset = NoiseOrderedDataset(
            nested_dataset,
            sort_order=[shuffle, src_dataset.sizes],
            seed=self.seed,
            order_noise=self.order_noise,
        )
        return dataset

    def build_model(self, cfg):
        model = super().build_model(cfg)
        model.register_classification_head(
            getattr(cfg, "classification_head_name", "sentence_classification_head"),
            num_classes=self.cfg.num_classes,
        )
        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary


class TruncateSizesDataset(BaseWrapperDataset):
    def __init__(self, dataset, truncation_length):
        super().__init__(dataset)
        self.truncation_length = truncation_length

    @property
    def sizes(self):
        return np.minimum(self.dataset.sizes, self.truncation_length)


class StandardScaler:
    def __init__(self, x):
        x= np.array(x).astype(np.float)
        self.means = np.nanmean(x, axis=0)
        self.stds = np.nanstd(x, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)
        self.means = float(self.means[0])
        self.stds = float(self.stds[0])

    def transform(self, x):
        return (x - self.means) / self.stds

    def inverse_transform(self, x):
        return x * self.stds + self.means

