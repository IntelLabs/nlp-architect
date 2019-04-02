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
import pickle
from os import makedirs, path, sys

import numpy as np

from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.models.ner_crf import NERCRF
from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import download_unlicensed_file
from nlp_architect.utils.text import SpacyInstance, bio_to_spans


class NerApi(AbstractApi):
    """
    NER model API
    """
    model_dir = str(LIBRARY_OUT / 'ner-pretrained')
    pretrained_model = path.join(model_dir, 'model_v4.h5')
    pretrained_model_info = path.join(model_dir, 'model_info_v4.dat')

    def __init__(self, prompt=True):
        self.model = None
        self.model_info = None
        self.word_vocab = None
        self.y_vocab = None
        self.char_vocab = None
        self._download_pretrained_model(prompt)
        self.nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])

    @staticmethod
    def _prompt():
        response = input('\nTo download \'{}\', please enter YES: '.
                         format('ner'))
        res = response.lower().strip()
        if res == "yes" or (len(res) == 1 and res == 'y'):
            print('Downloading {}...'.format('ner'))
            responded_yes = True
        else:
            print('Download declined. Response received {} != YES|Y. '.format(res))
            responded_yes = False
        return responded_yes

    def _download_pretrained_model(self, prompt=True):
        """Downloads the pre-trained BIST model if non-existent."""
        model_exists = path.isfile(self.pretrained_model)
        model_info_exists = path.isfile(self.pretrained_model_info)
        if not model_exists or not model_info_exists:
            print('The pre-trained models to be downloaded for the NER dataset '
                  'are licensed under Apache 2.0. By downloading, you accept the terms '
                  'and conditions provided by the license')
            makedirs(self.model_dir, exist_ok=True)
            if prompt is True:
                agreed = NerApi._prompt()
                if agreed is False:
                    sys.exit(0)
            download_unlicensed_file('https://s3-us-west-2.amazonaws.com/nlp-architect-data'
                                     '/models/ner/',
                                     'model_v4.h5', self.pretrained_model)
            download_unlicensed_file('https://s3-us-west-2.amazonaws.com/nlp-architect-data'
                                     '/models/ner/',
                                     'model_info_v4.dat', self.pretrained_model_info)
            print('Done.')

    def load_model(self):
        self.model = NERCRF()
        self.model.load(self.pretrained_model)
        with open(self.pretrained_model_info, 'rb') as fp:
            model_info = pickle.load(fp)
        self.word_vocab = model_info['word_vocab']
        self.y_vocab = {v: k for k, v in model_info['y_vocab'].items()}
        self.char_vocab = model_info['char_vocab']

    @staticmethod
    def pretty_print(text, tags):
        spans = []
        for s, e, tag in bio_to_spans(text, tags):
            spans.append({
                'start': s,
                'end': e,
                'type': tag
            })
        ents = dict((obj['type'].lower(), obj) for obj in spans).keys()
        ret = {'doc_text': ' '.join(text),
               'annotation_set': list(ents),
               'spans': spans,
               'title': 'None'}
        print({"doc": ret, 'type': 'high_level'})
        return {"doc": ret, 'type': 'high_level'}

    def process_text(self, text):
        input_text = ' '.join(text.strip().split())
        return self.nlp.tokenize(input_text)

    def vectorize(self, doc, vocab, char_vocab):
        words = np.asarray([vocab[w.lower()] if w.lower() in vocab else 1 for w in doc]) \
            .reshape(1, -1)
        sentence_chars = []
        for w in doc:
            word_chars = []
            for c in w:
                if c in char_vocab:
                    _cid = char_vocab[c]
                else:
                    _cid = 1
                word_chars.append(_cid)
            sentence_chars.append(word_chars)
        sentence_chars = np.expand_dims(pad_sentences(sentence_chars, self.model.word_length),
                                        axis=0)
        return words, sentence_chars

    def inference(self, doc):
        text_arr = self.process_text(doc)
        doc_vec = self.vectorize(text_arr, self.word_vocab, self.char_vocab)
        seq_len = np.array([len(text_arr)]).reshape(-1, 1)
        inputs = list(doc_vec)
        # pylint: disable=no-member
        inputs = list(doc_vec) + [seq_len]
        doc_ner = self.model.predict(inputs, batch_size=1).argmax(2).flatten()
        tags = [self.y_vocab.get(n, None) for n in doc_ner]
        return self.pretty_print(text_arr, tags)
