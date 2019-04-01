# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
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
from os import path
from pathlib import Path

from nlp_architect import LIBRARY_OUT

ABSA_ROOT = Path(path.realpath(__file__)).parent

TRAIN_LEXICONS = ABSA_ROOT / 'train' / 'lexicons'

TRAIN_CONF = ABSA_ROOT / 'train' / 'config.ini'

TRAIN_OUT = LIBRARY_OUT / 'absa' / 'train'

INFERENCE_LEXICONS = ABSA_ROOT / 'inference' / 'lexicons'

INFERENCE_OUT = LIBRARY_OUT / 'absa' / 'inference'
