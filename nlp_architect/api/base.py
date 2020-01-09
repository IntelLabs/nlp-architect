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
from typing import Union, List, Dict

logger = logging.getLogger(__name__)


class ModelAPI:
    """ Base class for a model API implementation
        Implementing classes must provide a default model and/or a path to a model

        Args:
            model_path (str): path to a trained model

        run method must return
    """

    default_model = None  # pre-trained model from library

    def __init__(self, model_path: str = None):
        if model_path is not None:
            self.load_model(model_path)
        elif self.default_model is not None:
            # get default model and load it
            # TODO: implement model integration
            raise NotImplementedError
        else:
            logger.error("Not model provided or not pre-trained model configured")

    def load_model(self, model_path: str):
        raise NotImplementedError

    def run(self, inputs: Union[str, List[str]]) -> Dict:
        raise NotImplementedError

    def __call__(self, inputs: Union[str, List[str]]):
        return self.run(inputs)
