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
import logging
import sys
from os import path, makedirs

from tqdm import tqdm

from nlp_architect.pipelines.spacy_np_annotator import NPAnnotator, get_noun_phrases
from nlp_architect.solutions.trend_analysis.scoring_utils import TextSpanScoring
from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.io import download_unlicensed_file
from nlp_architect.utils.text import SpacyInstance

nlp_chunker_url = 'https://s3-us-west-2.amazonaws.com/nlp-architect-data/models/chunker/'
chunker_model_dat_file = 'model_info.dat.params'
chunker_model_file = 'model.h5'
chunker_local_path = str(LIBRARY_OUT / 'chunker-pretrained')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class NPScorer(object):
    def __init__(self, parser=None):
        if parser is None:
            self.nlp = SpacyInstance(
                disable=['ner', 'parser', 'vectors', 'textcat']).parser
        else:
            self.nlp = parser

        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'), first=True)
        _path_to_model = path.join(chunker_local_path, chunker_model_file)
        if not path.exists(chunker_local_path):
            makedirs(chunker_local_path)
        if not path.exists(_path_to_model):
            logger.info(
                'The pre-trained model to be downloaded for NLP Architect word'
                ' chunker model is licensed under Apache 2.0')
            download_unlicensed_file(nlp_chunker_url, chunker_model_file, _path_to_model)
        _path_to_params = path.join(chunker_local_path, chunker_model_dat_file)
        if not path.exists(_path_to_params):
            download_unlicensed_file(nlp_chunker_url, chunker_model_dat_file, _path_to_params)
        self.nlp.add_pipe(NPAnnotator.load(_path_to_model, _path_to_params), last=True)

    def score_documents(self, texts: list, limit=-1, return_all=False, min_tf=5):
        documents = []
        assert len(texts) > 0, 'texts should contain at least 1 document'
        assert min_tf > 0, 'min_tf should be at least 1'
        with tqdm(total=len(texts), desc='documents scoring progress', unit='docs') as pbar:
            for doc in self.nlp.pipe(texts, n_threads=-1):
                if len(doc) > 0:
                    documents.append(doc)
                pbar.update(1)

        corpus = []
        for doc in documents:
            spans = get_noun_phrases(doc)
            if len(spans) > 0:
                corpus.append((doc, spans))

        if len(corpus) < 1:
            return []

        documents, doc_phrases = list(zip(*corpus))
        scorer = TextSpanScoring(documents=documents, spans=doc_phrases, min_tf=min_tf)
        tfidf_scored_list = scorer.get_tfidf_scores()
        if len(tfidf_scored_list) < 1:
            return []
        cvalue_scored_list = scorer.get_cvalue_scores()
        freq_scored_list = scorer.get_freq_scores()

        if limit > 0:
            tf = {tuple(k[0]): k[1] for k in tfidf_scored_list}
            cv = {tuple(k[0]): k[1] for k in cvalue_scored_list}
            fr = {tuple(k[0]): k[1] for k in freq_scored_list}
            tfidf_scored_list_limit = []
            cvalue_scored_list_limit = []
            freq_scored_list_limit = []
            for phrase in list(zip(*tfidf_scored_list))[0][:limit]:
                tfidf_scored_list_limit.append((phrase, tf[tuple(phrase)]))
                cvalue_scored_list_limit.append((phrase, cv[tuple(phrase)]))
                freq_scored_list_limit.append((phrase, fr[tuple(phrase)]))
            tfidf_scored_list = tfidf_scored_list_limit
            cvalue_scored_list = cvalue_scored_list_limit
            freq_scored_list = freq_scored_list_limit

        tfidf_scored_list = scorer.normalize_l2(tfidf_scored_list)
        cvalue_scored_list = scorer.normalize_l2(cvalue_scored_list)
        freq_scored_list = scorer.normalize_minmax(freq_scored_list, invert=True)
        tfidf_scored_list = scorer.normalize_minmax(tfidf_scored_list)
        cvalue_scored_list = scorer.normalize_minmax(cvalue_scored_list)
        if return_all:
            tf = {tuple(k[0]): k[1] for k in tfidf_scored_list}
            cv = {tuple(k[0]): k[1] for k in cvalue_scored_list}
            fr = {tuple(k[0]): k[1] for k in freq_scored_list}
            final_list = []
            for phrases in tf.keys():
                final_list.append(([p for p in phrases], tf[phrases], cv[phrases], fr[phrases]))
            return final_list
        merged_list = scorer.interpolate_scores([tfidf_scored_list, cvalue_scored_list],
                                                [0.5, 0.5])
        merged_list = scorer.multiply_scores([merged_list, freq_scored_list])
        merged_list = scorer.normalize_minmax(merged_list)
        final_list = []
        for phrases, score in merged_list:
            if any([len(p) > 1 for p in phrases]):
                final_list.append(([p for p in phrases], score))
        return final_list
