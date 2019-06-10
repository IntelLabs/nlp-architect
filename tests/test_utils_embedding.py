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

import numpy as np

from nlp_architect.utils.embedding import FasttextEmbeddingsModel
from nlp_architect.utils.testing import NLPArchitectTestCase

texts = ['The quick brown fox jumped over the fence',
         'NLP Architect is an open source library',
         'NLP Architect is made by Intel AI',
         'python is the scripting language used in NLP Architect']


class TestFasttextEmbeddingModel(NLPArchitectTestCase):
    def setUp(self):
        super().setUp()
        self.data = [t.split() for t in texts]
        self.model = FasttextEmbeddingsModel()
        self.file_path = str(self.TEST_DIR / 'fasttext_emb_model')

    def test_train(self):
        self.model.train(self.data, epochs=50)
        assert self.model

    def test_query(self):
        wv = self.model['NLP']
        assert wv is not None
        assert isinstance(wv, np.ndarray)

    def test_save_load(self):
        self.model.train(self.data, epochs=50)
        self.model.save(self.file_path)
        new_model = FasttextEmbeddingsModel.load(self.file_path)
        assert new_model is not None
        assert isinstance(new_model['word'], np.ndarray)
