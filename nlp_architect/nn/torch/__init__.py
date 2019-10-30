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
import random
import time

import numpy as np
import torch


def setup_backend(no_cuda):
    """Setup backend according to selected backend and detected configuration
    """
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    if torch.cuda.is_available() and not no_cuda:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        n_gpu = 0
    return device, n_gpu


def set_seed(seed, n_gpus=None):
    """set and return seed
    """
    if seed == -1:
        seed = int(time.time())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpus is not None and n_gpus > 0:
        torch.cuda.manual_seed_all(seed)
    return seed
