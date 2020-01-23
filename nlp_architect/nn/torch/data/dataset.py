# ******************************************************************************
# Copyright 2017-2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import torch
from typing import List


class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


class ConcatTensorDataset(torch.utils.data.Dataset):
    r"""Dataset as a concatenation of multiple TensorDataset datasets.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        dataset (TensorDataset): dataset to which rest datasets will be concatinated.
        datasets (List[TensorDataset]): datasets to concat to the dataset.
    """

    def __init__(
            self,
            dataset: torch.utils.data.TensorDataset,
            datasets: List[torch.utils.data.TensorDataset]):
        tensors = dataset.tensors
        for ds in datasets:
            concat_tensors = []
            for i, t in enumerate(ds.tensors):
                concat_tensors.append(torch.cat((tensors[i], t), 0))
            tensors = concat_tensors
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
