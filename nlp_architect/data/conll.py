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
import re


class ConllEntry:
    def __init__(self, eid, form, lemma, pos, cpos, feats=None, parent_id=None,
                 relation=None, deps=None, misc=None):
        self.id = eid
        self.form = form
        self.norm = normalize(form)
        self.cpos = cpos.upper()
        self.pos = pos.upper()
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None

        self.vec = None
        self.lstms = None

    def __str__(self):
        values = [str(self.id), self.form, self.lemma, self.cpos, self.pos,
                  self.feats,
                  str(self.pred_parent_id) if self.pred_parent_id is not None
                  else None,
                  self.pred_relation, self.deps,
                  self.misc]
        return '\t'.join(['_' if v is None else v for v in values])


def normalize(word):
    return 'NUM' if NUMBER_REGEX.match(word) else word.lower()


NUMBER_REGEX = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")