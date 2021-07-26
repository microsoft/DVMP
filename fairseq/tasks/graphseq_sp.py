import logging
import os
import numpy as np
from fairseq.data import(
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    OffsetTokensDataset,
    StripTokenDataset,
    NumSamplesDataset,
    NumelDataset,
    data_utils,
    LeftPadDataset,
    BaseWrapperDataset
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import FairseqTask, register_task
from fairseq.dataclass import FairseqDataclass, ChoiceEnum
from dataclasses import dataclass, field
from typing import Optional, List
from omegaconf import II
from fairseq.data.indexed_dataset import (get_available_dataset_impl,
                                          make_dataset,
                                          infer_dataset_impl)
from fairseq.data.molecule.indexed_dataset import MolMMapIndexedDataset
from fairseq.data.molecule.indexed_dataset import make_dataset as make_graph_dataset
from torch_geometric.data import Data, Batch


logger = logging.getLogger(__name__)


@dataclass
class GraphSeqPredictionConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None,
        metadata={
            "help": "colon separated path to data directories list, will be iterated upon during epochs "
                    "in round-robin manner; however, valid and test data are always in the first directory "
                    "to avoid the need for repeating them in all directories"
        }
    )
    num_classes: int = field(default=2)
    regression_target: bool = field(default=False)
    no_shuffle: bool = field(default=False)
    shorten_method: ChoiceEnum(["none", "truncate", "random_crop"]) = field(
        default="truncate"
    )
    shorten_data_split_list: str = field(default='')
    max_positions: int = II("model.max_positions")
    dataset_impl: Optional[ChoiceEnum(get_available_dataset_impl())] = II(
        "dataset.dataset_impl"
    )
    seed: int = II("common.seed")
    order_noise: int = field(default=5)


@register_task("graphseq_sp", dataclass=GraphSeqPredictionConfig)
class GraphSeqPredicition(FairseqTask):

    cfg: GraphSeqPredictionConfig

    def __init__(self, cfg: GraphSeqPredictionConfig,
                 data_dictionary,
                 label_dictionary):
        super().__init__(cfg)
        self.dictionary = data_dictionary
        self.dictionary.add_symbol("[MASK]")
        self._label_dictionary = label_dictionary
        self._max_positions = cfg.max_positions
        self.seed = cfg.seed
        self.order_noise = cfg.order_noise

    @classmethod
    def setup_task(cls, cfg: GraphSeqPredictionConfig, **kwargs):
        assert cfg.num_classes > 0
        data_dict = cls.load_dictionary(os.path.join(
            cfg.data, 'input0', "dict.txt"
        ))
        logger.info("[input] Dictionary {}: {} types.".format(
            os.path.join(cfg.data, "input0",),
            len(data_dict)
        ))
        if not cfg.regression_target:
            label_dict = cls.load_dictionary(os.path.join(
                cfg.data, "label", "dict.txt"
            ))
            logger.info("[label] Dictionary {}: {} types.".format(
                os.path.join(cfg.data, "label"),
                len(label_dict)
            ))
        else:
            label_dict = data_dict
        return cls(cfg, data_dict, label_dict)

    def load_dataset(
        self,
        split: str,
        combine: bool = False,
        **kwargs
    ):

        def get_path(key, split):
            return os.path.join(self.cfg.data, key, split)

        prefix = get_path('input0', split)
        if not MolMMapIndexedDataset.exists(prefix):
            raise FileNotFoundError("Graph data {} not found.".format(prefix))
        if self.cfg.dataset_impl is None:
            dataset_impl = infer_dataset_impl(prefix)
        else:
            dataset_impl = self.cfg.dataset_impl

        seq_dataset = make_dataset(prefix, impl=dataset_impl)
        assert seq_dataset is not None
        graph_dataset = make_graph_dataset(prefix, impl=dataset_impl)
        assert graph_dataset is not None

        with data_utils.numpy_seed(self.cfg.seed):
            shuffle = np.random.permutation(len(seq_dataset))

        seq_dataset = maybe_shorten_dataset(
            seq_dataset,
            split,
            self.cfg.shorten_data_split_list,
            self.cfg.shorten_method,
            self._max_positions,
            self.cfg.seed
        )
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": LeftPadDataset(
                    seq_dataset,
                    pad_idx=self.dictionary.pad()
                ),
                "src_lengths": NumelDataset(seq_dataset, reduce=False)
            },
            "graph": GraphCollateDataset(graph_dataset),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(seq_dataset, reduce=True)
        }
        if not self.cfg.regression_target:
            prefix = get_path('label', split)
            label_dataset = make_dataset(prefix, impl=dataset_impl)
            assert label_dataset is not None
            dataset.update(
                target=OffsetTokensDataset(
                    StripTokenDataset(
                        label_dataset,
                        id_to_strip=self.label_dictionary.eos()
                    ),
                    offset=-self.label_dictionary.nspecial
                )
            )
        else:
            raise NotImplementedError
        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[seq_dataset.sizes]
        )

        if self.cfg.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = NoiseOrderedDataset(
                nested_dataset,
                sort_order=[shuffle],
                seed=self.seed,
                order_noise=self.order_noise
            )

        logger.info("Loaded {} with #samples: {}.".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]


    def build_model(self, cfg):
        model = super().build_model(cfg)
        model.register_classification_head(
            getattr(cfg, "classification_head_name", "sentence_classification_head"),
            num_classes=self.cfg.num_classes
        )
        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def src_dict(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary


class NoiseOrderedDataset(BaseWrapperDataset):

    def __init__(self, dataset, sort_order,
                 seed, order_noise):
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
                    so + np.random.randint(low=-self.order_noise,
                                           high=self.order_noise,
                                           size=so.shape)
                )
            return np.lexsort(self.sort_order)

    def set_epoch(self, epoch):
        self._epoch = epoch
        super().set_epoch(epoch)


class GraphCollateDataset(BaseWrapperDataset):

    def __init__(self, dataset):
        super().__init__(dataset)

    def collater(self, samples):
        data_list = []
        for s in samples:
            data_list.append(Data(
                x=s['node_attr'],
                edge_index=s['edge_index'].T,
                edge_attr=s['edge_attr'],
            ))
        return Batch.from_data_list(data_list)
