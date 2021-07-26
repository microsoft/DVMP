# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import shutil
import struct
import os
from functools import lru_cache
import numpy as np
import torch
from fairseq.file_io import PathManager
from fairseq.data.indexed_dataset import (_code_to_dtype,
                                          _warmup_mmap_file,
                                          _dtype_header_code,
                                          MMapIndexedDataset,
                                          index_file_path,
                                          data_file_path,
                                          best_fitting_int_dtype)
from molecule.features import get_bond_feature_dims, get_atom_feature_dims


def make_builder(out_file, impl, vocab_size=None):
    if impl == "mmap":
        return MolMMapIndexedDatasetBuilder(out_file,
                                            dtype=best_fitting_int_dtype(vocab_size))
    else:
        raise NotImplementedError()


def make_dataset(path, impl):
    if impl == "mmap":
        return MolMMapIndexedDataset(path)
    else:
        raise NotImplementedError()


class TwoDimMMapIndexedDataset(MMapIndexedDataset):
    class Index:
        _HDR_MAGIC = b"TwoDMMIDIDX\x00\x00"

        @classmethod
        def writer(cls, path, dtype, dim):
            class _Writer:
                def __enter__(self):
                    self._file = open(path, 'wb')
                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack("<Q", 1))
                    self._file.write(struct.pack("<Q", dim))
                    self._file.write(struct.pack("<B", _dtype_header_code(dtype)))
                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []
                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size
                    return pointers

                def write(self, sizes):
                    pointers = self._get_pointers(sizes)
                    self._file.write(struct.pack("<Q", len(sizes)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order="C"))
                    del pointers

                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path):
            with open(path, 'rb') as stream:
                magic_test = stream.read(13)
                assert self._HDR_MAGIC == magic_test, (
                    "Index file does not math expected format."
                    "Make sure that --dataset-impl is configured properly."
                )
                version = struct.unpack("<Q", stream.read(8))
                assert (1,) == version

                self.dim = struct.unpack("<Q", stream.read(8))[0]
                dtype_code = struct.unpack("<B", stream.read(1))[0]
                self._dtype = _code_to_dtype[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack("<Q", stream.read(8))[0]
                offset = stream.tell()

            _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order="C")
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            self._sizes = np.frombuffer(
                self._bin_buffer, dtype=np.int32, count=self._len, offset=offset
            )
            self._pointers = np.frombuffer(
                self._bin_buffer,
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes,
            )

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, path):
        super().__init__(path)

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        item = super().__getitem__(i)
        item = item.reshape(-1, self._index.dim)
        return item


class MolMMapIndexedDataset(torch.utils.data.Dataset):

    def __init__(self, path):
        super().__init__()
        self._path = None
        self.datasets = None
        self._do_init(path)

    @staticmethod
    def attrs():
        return [
            'node_attr',
            'edge_attr',
            'edge_index',
            'num_nodes'
        ]

    @staticmethod
    def attrs2dim():
        return [
            len(get_atom_feature_dims()),
            len(get_bond_feature_dims()),
            2,
            1
        ]

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    @staticmethod
    def get_attr_fns(fn):
        prefix, suffix = os.path.splitext(fn)
        return ['{}.{}{}'.format(prefix, x, suffix)
                for x in MolMMapIndexedDataset.attrs()]

    def _do_init(self, path):
        self._path = path
        self.datasets = []
        for attr in self.attrs():
            self.datasets.append(TwoDimMMapIndexedDataset('{}.{}'.format(path, attr)))
        _size = self.datasets[self.attrs().index('node_attr')].sizes / len(get_atom_feature_dims())
        self._size = _size.astype(self.datasets[self.attrs().index('node_attr')].sizes.dtype)

    def __del__(self):
        for dataset in self.datasets:
            del dataset

    def __len__(self):
        return len(self.datasets[0])

    @lru_cache(maxsize=8)
    def __getitem__(self, item):
        ret_dict = {}
        for attr, dataset in zip(self.attrs(), self.datasets):
            ret_dict[attr] = dataset[item]
        return ret_dict

    @property
    def sizes(self):
        return self._size

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        exist = True
        for index_fn, data_fn in zip(MolMMapIndexedDataset.get_attr_fns(index_file_path(path)),
                                     MolMMapIndexedDataset.get_attr_fns(data_file_path(path))):
            exist = exist and PathManager.exists(index_fn) and PathManager.exists(data_fn)
        return exist


class MolMMapIndexedDatasetBuilder:
    def __init__(self, out_file, dtype=np.int64):
        self.mol_attrs = MolMMapIndexedDataset.attrs()
        _data_fns = self.get_attr_fns(out_file)
        self._data_files = [open(fn, 'wb') for fn in _data_fns]
        self._dtype = dtype
        self._sizes = [[] for attr in self.mol_attrs]

    def add_item(self, graph):
        assert len(graph) == len(self.mol_attrs)
        for i, attr in enumerate(self.mol_attrs):
            np_array = np.array(graph[attr], dtype=self._dtype)
            self._data_files[i].write(np_array.tobytes(order="C"))
            self._sizes[i].append(np_array.size)

    def merge_file_(self, another_file):
        index_fns = index_file_path(another_file)
        another_fns = self.get_attr_fns(index_fns)
        for i, fn in enumerate(another_fns):
            index = TwoDimMMapIndexedDataset.Index(fn)
            assert index.dtype == self._dtype
            for size in index.sizes:
                self._sizes[i].append(size)

            with open(self.get_attr_fns(data_file_path(another_file))[i], "rb") as f:
                shutil.copyfileobj(f, self._data_files[i])

    def get_attr_fns(self, fn):
        prefix, suffix = os.path.splitext(fn)
        return ['{}.{}{}'.format(prefix, x, suffix)
                for x in self.mol_attrs]

    def finalize(self, index_file):
        for f in self._data_files:
            f.close()

        _data_fns = self.get_attr_fns(index_file)
        for i, dim in enumerate(MolMMapIndexedDataset.attrs2dim()):
            index_file = _data_fns[i]
            sizes = self._sizes[i]
            with TwoDimMMapIndexedDataset.Index.writer(index_file, self._dtype, dim) as index:
                index.write(sizes)

    def remove_temp_files(self, another_file):
        index_fns = index_file_path(another_file)
        another_fns = self.get_attr_fns(index_fns)
        for fn in another_fns:
            os.remove(fn)
        data_fns = data_file_path(another_file)
        another_fns = self.get_attr_fns(data_fns)
        for fn in another_fns:
            os.remove(fn)

