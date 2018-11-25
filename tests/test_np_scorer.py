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
from nlp_architect.solutions.trend_analysis.np_scorer import NPScorer
from nlp_architect.utils.io import load_files_from_path


def test_np_scorer():
    doc1 = 'It is based on representing any term of a training corpus using word embeddings in ' \
           'order to estimate the similarity between the seed terms and any candidate term.' \
           ' Noun phrases provide good approximation for candidate terms and are extracted in' \
           'our system using a noun phrase chunker. At expansion time, given a seed of terms, ' \
           'the most similar terms are returned.'
    doc2 = 'The expand server gets requests containing seed terms, and expands them based on' \
           ' the given word embedding model. You can use the model you trained yourself in the ' \
           'previous step, or to provide a pre-trained model you own. Important note: ' \
           'default server will listen on localhost:1234. If you set the host/port you ' \
           'should also set it in the ui/settings.py file.'
    docs = [doc1, doc2]
    np = NPScorer()
    phrase_scores = np.score_documents(docs, limit=3, return_all=True, min_tf=1)
    assert len(phrase_scores) == 3
    phrase_scores = np.score_documents(docs, return_all=True, min_tf=100)
    assert len(phrase_scores) == 0


def test_extract_and_score(file_path=None):
    if file_path:
        documents = load_files_from_path(file_path)
        np = NPScorer()
        phrases = np.score_documents(documents, return_all=True, min_tf=1)
        assert phrases
