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
# pylint: disable=no-name-in-module
import itertools
import math

import numpy as np
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token
from wordfreq import zipf_frequency


class TextSpanScoring:
    """
    Text spans scoring class.
    Contains misc scoring algorithms for scoring text fragments extracted
    from a corpus.

    Arguments:
        documents(list): List of spaCy documents.
        spans(list[list]): List of spaCy spans representing noun phrases of documents
            document.
    """

    def __init__(self, documents, spans, min_tf=1):
        assert len(documents) == len(spans)
        self._documents = documents
        self._doc_text_spans = spans
        self.index = CorpusIndex(documents, spans)
        assert min_tf > 0, 'min_tf must be > 0'
        if min_tf > 1:
            self.re_index_min_tf(min_tf)

    def re_index_min_tf(self, tf):
        filtered_doc_text_spans = []
        for d in self.doc_text_spans:
            filtered_doc_phrases = [p for p in d if self.index.tf(p) >= tf]
            filtered_doc_text_spans.append(filtered_doc_phrases)
        self._doc_text_spans = filtered_doc_text_spans
        self.index = CorpusIndex(self.documents, self.doc_text_spans)

    @property
    def documents(self):
        return self._documents

    @property
    def doc_text_spans(self):
        return self._doc_text_spans

    def get_tfidf_scores(self, group_similar_spans=True):
        """
        Get TF-IDF scores of spans
        return a list of spans sorted by desc order of importance
        span score = TF (global) * (1 + log_n(DF/N))
        """
        phrases_and_scores = {}
        num_of_docs = len(self.documents)
        for _, noun_phrases in zip(self.documents, self.doc_text_spans):
            for p in noun_phrases:
                if p not in phrases_and_scores:
                    tf = self.index.tf(p)
                    df = self.index.df(p)
                    phrases_and_scores[p] = \
                        (tf + 1) * math.log(1 + num_of_docs / df)
        if len(phrases_and_scores) > 0:
            return self._maybe_group_and_sort(group_similar_spans,
                                              phrases_and_scores)
        return []

    def get_freq_scores(self, group_similar_spans=True):
        phrases_and_scores = {}
        for _, noun_phrases in zip(self.documents, self.doc_text_spans):
            for p in noun_phrases:
                if p not in phrases_and_scores:
                    phrases_and_scores[p] = zipf_frequency(p.text, 'en')
        return self._maybe_group_and_sort(group_similar_spans,
                                          phrases_and_scores)

    def group_spans(self, phrases):
        pid_phrase_scores = [{'k': self.index.get_pid(p),
                              'v': (p, s)}
                             for p, s in phrases.items()]
        phrase_groups = []
        for _, group in itertools.groupby(
                sorted(pid_phrase_scores, key=lambda x: x['k']),
                lambda x: x['k']):
            _group = list(group)
            phrases = {g['v'][0].text for g in _group}
            score = _group[0]['v'][1]
            phrase_groups.append((sorted(phrases), score))
        return phrase_groups

    def _maybe_group_and_sort(self, is_group, phrases_dict):
        phrase_groups = phrases_dict.items()
        if is_group:
            phrase_groups = self.group_spans(phrases_dict)
        return sorted(phrase_groups, key=lambda x: x[1], reverse=True)

    @staticmethod
    def normalize_minmax(phrases_list, invert=False):
        _, scores = list(zip(*phrases_list))
        max_score = max(scores)
        min_score = min(scores)
        norm_list = []
        for p, s in phrases_list:
            if max_score - min_score > 0:
                new_score = (s - min_score) / (max_score - min_score)
            else:
                new_score = 0
            norm_list.append([p, new_score])
        if invert:
            for e in norm_list:
                e[1] = 1.0 - e[1]
        return norm_list

    @staticmethod
    def normalize_l2(phrases_list):
        phrases, scores = list(zip(*phrases_list))
        scores = scores / np.linalg.norm(scores)
        return list(zip(phrases, scores.tolist()))

    @staticmethod
    def interpolate_scores(phrase_lists, weights=None):
        if weights is None:
            weights = [1.0 / len(phrase_lists)]
        else:
            assert len(weights) == len(phrase_lists)

        list_sizes = [len(l) for l in phrase_lists]
        for l in list_sizes:
            assert len(phrase_lists[0]) == l, 'list sizes not equal'

        phrase_list_dicts = []
        for lst in phrase_lists:
            phrase_list_dicts.append({tuple(k): v for k, v in lst})

        phrases = phrase_list_dicts[0].keys()
        interp_scores = {}
        for p in phrases:
            interp_scores[p] = 0.0
            for ref, w in zip(phrase_list_dicts, weights):
                interp_scores[p] += ref[p] * w
        return sorted(interp_scores.items(), key=lambda x: x[1], reverse=True)

    def get_cvalue_scores(self, group_similar_spans=True):
        phrases_text = [p for ps in self.index.pid_to_spans.values() for p in ps]
        phrase_scores = {}
        for phrase in phrases_text:
            sub_phrase_pids = None
            for w in phrase:
                sub_pids = self.index.get_subphrases_of_word(w)
                if sub_phrase_pids is None:
                    sub_phrase_pids = sub_pids
                else:
                    sub_phrase_pids = sub_phrase_pids.intersection(sub_pids)
            sub_phrase_pids.discard(self.index.get_pid(phrase))

            if len(sub_phrase_pids) > 0:
                score = math.log2(1 + len(phrase)) * (
                    self.index.tf(phrase) - 1.0 / len(sub_phrase_pids) * sum(
                        [self.index.tf(list(self.index.get_phrase(p))[0])
                         for p in sub_phrase_pids]))
            else:
                score = math.log2(1 + len(phrase)) * self.index.tf(phrase)
            phrase_scores[phrase] = score
        for phrase in phrase_scores:
            score = [phrase_scores[p] for p in
                     self.index.get_phrase(self.index.get_pid(phrase))]
            phrase_scores[phrase] = sum(score) / len(score)
        return self._maybe_group_and_sort(group_similar_spans,
                                          phrase_scores)

    @staticmethod
    def multiply_scores(phrase_lists):
        phrase_list_dicts = []
        for lst in phrase_lists:
            phrase_list_dicts.append({tuple(k): v for k, v in lst})
        phrases = phrase_list_dicts[0].keys()
        interp_scores = {}
        for p in phrases:
            interp_scores[p] = 1.0
            for ref in phrase_list_dicts:
                interp_scores[p] *= ref[p]
        return sorted(interp_scores.items(), key=lambda x: x[1], reverse=True)


class CorpusIndex:
    """
    Text span index class.
    Holds TF and DF values per span. Text spans are normalized and similar
    spans are mapped to the same TF DF values.
    """

    def __init__(self, documents: list, spans: list):
        self.df_index = {}  # span to DF
        self.tf_index = {}  # span, doc to TF
        self.pid_to_spans = {}  # id to spans (normalized types)
        self.word_to_phrase_index = {}
        self.documents = set()
        for d, phrases in zip(documents, spans):
            doc_id = CorpusIndex.get_docid(d)
            self.documents.add(doc_id)
            for phrase in phrases:
                pid = CorpusIndex.get_pid(phrase)
                # add to norm phrases dict
                if pid not in self.pid_to_spans:
                    self.pid_to_spans[pid] = set()
                self.pid_to_spans[pid].add(phrase)

                # add to df index
                if self.df_index.get(pid, None) is None:
                    self.df_index[pid] = {doc_id}
                else:
                    self.df_index[pid].add(doc_id)

                # add to tf index
                if self.tf_index.get(pid, None) is None:
                    self.tf_index[pid] = 1
                else:
                    self.tf_index[pid] += 1

                for word in phrase:
                    wid = self.get_wid(word)
                    if wid not in self.word_to_phrase_index:
                        self.word_to_phrase_index[wid] = set()
                    self.word_to_phrase_index[wid].add(pid)

    def get_phrase(self, pid):
        return self.pid_to_spans.get(pid)

    def get_subphrases_of_word(self, w):
        wid = self.get_wid(w)
        return self.word_to_phrase_index.get(wid)

    @staticmethod
    def get_wid(w: Token):
        """
        get word id
        """
        return CorpusIndex.hash_func(w.text)

    @staticmethod
    def get_pid(p: Span):
        """
        get phrase id
        """
        return CorpusIndex.hash_func(p.lemma_)

    @staticmethod
    def get_docid(d: Doc):
        """
        get doc id
        """
        return CorpusIndex.hash_func(d)

    @staticmethod
    def hash_func(x):
        return hash(x)

    def tf(self, phrase):
        """
        Get TF of phrase in doc
        """
        pid = CorpusIndex.get_pid(phrase)
        if self.tf_index.get(pid, None) is not None:
            return self.tf_index.get(pid)
        return 0

    def df(self, phrase):
        """
        Get DF of phrase in corpus
        """
        pid = CorpusIndex.get_pid(phrase)
        if self.df_index.get(pid, None) is not None:
            return len(self.df_index[pid])
        return 0
