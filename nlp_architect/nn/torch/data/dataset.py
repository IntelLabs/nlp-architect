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
    r"""Dataset as a concatenation of multiple TensorDataset datasets with same number of tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        dataset (TensorDataset): dataset to which rest datasets will be concatinated.
        datasets (List[TensorDataset]): datasets to concat to the dataset.
    """

    def __init__(
        self,
        dataset: torch.utils.data.TensorDataset,
        datasets: List[torch.utils.data.TensorDataset],
    ):
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


class CombinedTensorDataset(torch.utils.data.Dataset):
    r"""Dataset as a concatenation of labeled dataset and unlabeled dataset. Labels of unlabeled dataset will
        be represented as zero tensors.

        Each sample will be retrieved by indexing tensors along the first dimension.

        Arguments:
            labeled_dataset (TensorDataset): labeled dataset
            unlabeled_datasets (TensorDataset): unlabeled dataset to concat
    """

    def __init__(
            self,
            labeled_dataset: torch.utils.data.TensorDataset,
            unlabeled_dataset: torch.utils.data.TensorDataset
        ):
        # max_ds_len = max([len(ds.tensors) for ds in datasets])
        new_tensors = ()
        # add 'zero labels' to the unlabeled dataset tensors
        unlbld_examples_count = unlabeled_dataset.tensors[0].shape[0]
        labeled_examples_shape = labeled_dataset.tensors[-1].shape
        empty_labels_shape = [unlbld_examples_count] + list(labeled_examples_shape[1:]) if len(labeled_examples_shape) > 1 else [unlbld_examples_count]
        unlabeled_dataset.tensors += (torch.tensor(torch.zeros(empty_labels_shape), dtype=int),)
        # concat
        for i in range(len(labeled_dataset.tensors)):
            new_tensors += (torch.cat([ds.tensors[i] for ds in [labeled_dataset, unlabeled_dataset]], dim=0),)
        assert all(new_tensors[0].size(0) == tensor.size(0) for tensor in new_tensors)
        self.tensors = new_tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].size(0)
