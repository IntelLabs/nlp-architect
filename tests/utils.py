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

import os


def count_conll_examples(file):
    ctr = 0
    if os.path.exists(file):
        with open(file) as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0:
                    ctr += 1
    else:
        print("File:" + file + " doesn't exist")
    return ctr
