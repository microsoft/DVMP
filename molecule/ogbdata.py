import os
import torch
import pandas as pd
from os import path as osp
from ogb.graphproppred import PygGraphPropPredDataset

class DatasetWrapper(PygGraphPropPredDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_idx_split(self, split_type=None):
        if split_type is None:
            split_type = self.meta_info['split']
        
        path = os.path.join(self.root, 'split', split_type)

        if os.path.isfile(os.path.join(path, 'split_dict.pt')):
            return torch.load(os.path.join(path, 'split_dict.pt'))

        train_idx = pd.read_csv(osp.join(path, 'train.csv.gz'), compression='gzip', header = None).values.T[0].tolist()
        valid_idx = pd.read_csv(osp.join(path, 'valid.csv.gz'), compression='gzip', header = None).values.T[0].tolist()
        test_idx = pd.read_csv(osp.join(path, 'test.csv.gz'), compression='gzip', header = None).values.T[0].tolist()
        return train_idx, valid_idx, test_idx

    def get_smiles(self):
        path = osp.join(self.root, 'mapping')
        smiles = pd.read_csv(osp.join(path, 'mol.csv.gz'), compression='gzip')['smiles'].values.tolist()
        return smiles 

    def get_labels(self):
        path = osp.join(self.root, 'raw')
        labels = pd.read_csv(osp.join(path, 'graph-label.csv.gz'), compression='gzip', header=None).values.tolist()
        return labels

    def get_dataset(self):
        train_idx, valid_idx, test_idx = self.get_idx_split()
        smiles = self.get_smiles()
        labels = self.get_labels()
        assert len(labels) == len(smiles)
        assert len(smiles) == len(set(train_idx)) +len(set(valid_idx)) + len(set(test_idx))
        assert len(smiles) == max(max(train_idx), max(valid_idx), max(test_idx)) + 1
        train_data = pd.DataFrame({
            "text": [smiles[i] for i in train_idx], "labels": [labels[i] for i in train_idx]
        })
        valid_data = pd.DataFrame({
            'text': [smiles[i] for i in valid_idx], 'labels': [labels[i] for i in valid_idx]
        })
        test_data = pd.DataFrame({
            'text': [smiles[i] for i in test_idx], 'labels': [labels[i] for i in test_idx]
        })
        return ['_'.join(self.name.split('-'))], [train_data, valid_data, test_data], None 

    def download(self):
        return super().download()

    def process(self):
        return super().process()