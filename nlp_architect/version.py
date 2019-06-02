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

MAJOR_V = 0
MINOR_V = 4
PATCH_V = 0
STAGE = 'post2'


def nlp_architect_version():
    if PATCH_V != 0:
        v = '{}.{}.{}'.format(MAJOR_V, MINOR_V, PATCH_V)
    else:
        v = '{}.{}'.format(MAJOR_V, MINOR_V)
    if len(STAGE) != 0:
        v += '.{}'.format(STAGE)
    return v


NLP_ARCHITECT_VERSION = nlp_architect_version()
