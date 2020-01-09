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
import logging
from abc import ABC

logger = logging.getLogger(__name__)


class TrainableModel(ABC):
    """Base class for a trainable model
    """

    def convert_to_tensors(self, *args, **kwargs):
        """convert any chosen input to valid model format of tensors
        """

    def get_logits(self, *args, **kwargs):
        """get model logits from given input
        """

    def train(self, *args, **kwargs):
        """train the model
        """

    def inference(self, *args, **kwargs):
        """run inference
        """

    def save_model(self, *args, **kwargs):
        """save the model
        """

    def load_model(self, *args, **kwargs):
        """load a model
        """
