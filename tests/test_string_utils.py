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
from nlp_architect.utils.string_utils import StringUtils


def test_is_determiner():
    assert StringUtils.is_determiner("the")
    assert StringUtils.is_determiner("on") is False


def test_is_preposition():
    assert StringUtils.is_preposition("the") is False
    assert StringUtils.is_preposition("on")


def test_is_pronoun():
    assert StringUtils.is_pronoun("anybody")
    assert StringUtils.is_pronoun("the") is False


def test_is_stopword():
    assert StringUtils.is_stop("always")
    assert StringUtils.is_stop("sunday") is False
