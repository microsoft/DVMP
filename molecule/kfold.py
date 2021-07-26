import os
import pandas as pd
from os import path as osp
import urllib.request as ur
import io
from collections import defaultdict
from tqdm import tqdm
from typing import Union
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import random


class KfoldDataset:
    def __init__(self, name: str):
        assert name.startswith("kfold_s")
        self.seed = int(name[len("kfold_s")])
        self.name = name.split("_")[-1]
        assert self.name in [
            "bace",
            "bbbp",
            "clintox",
            "sider",
            "tox21",
            "toxcast",
            "freesolv",
            "esol",
            "lipo",
            "qm7",
            "qm8",
        ]
        self.url = "https://raw.githubusercontent.com/tencent-ailab/grover/main/exampledata/finetune/{}.csv".format(
            self.name
        )
        self.download()

    def download(self):
        fndir = "/tmp/kfold"
        os.makedirs(fndir, exist_ok=True)
        fn = self.url.rpartition("/")[2]
        fn = osp.join(fndir, fn)
        self.fn = fn
        if os.path.exists(fn):
            return

        with ur.urlopen(self.url) as src:
            content = src.read().decode("utf8")
        with io.open(fn, "w", encoding="utf8") as tgt:
            tgt.write(content)

    def read_csv(self):
        return pd.read_csv(self.fn)

    def scaffold_split(self, data, sizes=(0.8, 0.1, 0.1), balanced: bool = True):
        assert sum(sizes) == 1
        train_size, val_size, test_size = (
            sizes[0] * len(data),
            sizes[1] * len(data),
            sizes[2] * len(data),
        )
        train, val, test = [], [], []
        train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0

        def scaffold2smiles(mols):
            scaffolds = defaultdict(set)
            for i, mol in tqdm(enumerate(mols), total=len(mols)):
                scaffold = generate_scaffold(mol)
                scaffolds[scaffold].add(i)
            return scaffolds

        scaffload2indices = scaffold2smiles(data)

        if balanced:
            index_sets = list(scaffload2indices.values())
            big_index_sets = []
            small_index_sets = []
            for index_set in index_sets:
                if len(index_set) > val_size / 2 or len(index_set) > test_size / 2:
                    big_index_sets.append(index_set)
                else:
                    small_index_sets.append(index_set)
            random.seed(self.seed)
            random.shuffle(big_index_sets)
            random.shuffle(small_index_sets)
            index_sets = big_index_sets + small_index_sets
        else:
            raise NotImplementedError()

        for index_set in index_sets:
            if len(train) + len(index_set) <= train_size:
                train += index_set
                train_scaffold_count += 1
            elif len(val) + len(index_set) <= val_size:
                val += index_set
                val_scaffold_count += 1
            else:
                test += index_set
                test_scaffold_count += 1

        return train, val, test

    def get_dataset(self):
        all_data = self.read_csv()
        if "smiles" in list(all_data.columns):
            all_smiles = all_data["smiles"].values.tolist()
        else:
            all_smiles = all_data["mol"].values.tolist()
        train_idx, valid_idx, test_idx = self.scaffold_split(all_smiles)
        all_tasks = list(all_data.columns)[1:]
        labels = all_data[all_tasks].values.tolist()

        def ret_pd(idx):
            smiles_list = [all_smiles[i] for i in idx]
            labels_split = [labels[i] for i in idx]
            return pd.DataFrame({"text": smiles_list, "labels": labels_split})

        pds = [ret_pd(train_idx), ret_pd(valid_idx), ret_pd(test_idx)]

        return all_tasks, pds, None


def generate_scaffold(mol: Union[str, Chem.Mol], include_chirality: bool = False) -> str:
    mol = Chem.MolFromSmiles(mol) if type(mol) == str else mol
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    return scaffold
