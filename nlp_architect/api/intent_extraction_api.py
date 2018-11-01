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

import numpy as np
import pickle
from os import makedirs, path, sys

from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.models.intent_extraction import MultiTaskIntentModel, Seq2SeqIntentModel
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import download_unlicensed_file
from nlp_architect.utils.text import SpacyInstance

nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])


class IntentExtractionApi(AbstractApi):
    dir = path.dirname(path.realpath(__file__))
    pretrained_model_info = path.join(dir, 'intent-pretrained', 'model_info.dat')
    pretrained_model = path.join(dir, 'intent-pretrained', 'model.h5')

    def __init__(self, prompt=True):
        self.model = None
        self.dir = path.dirname(path.realpath(__file__))
        self.model_info_path = IntentExtractionApi.pretrained_model_info
        self.model_path = IntentExtractionApi.pretrained_model
        self._download_pretrained_model(prompt)
        with open(self.model_info_path, 'rb') as fp:
            model_info = pickle.load(fp)
        self.model_type = model_info['type']
        self.word_vocab = model_info['word_vocab']
        self.tags_vocab = {v: k for k, v in model_info['tags_vocab'].items()}
        if self.model_type == 'mtl':
            self.char_vocab = model_info['char_vocab']
            self.intent_vocab = {v: k for k, v in model_info['intent_vocab'].items()}

    def process_text(self, text):
        input_text = ' '.join(text.strip().split())
        return nlp.tokenize(input_text)

    @staticmethod
    def _prompt():
        response = input('\nTo download \'{}\', please enter YES: '.
                         format('intent_extraction'))
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
        dir_path = path.join(self.dir, 'intent-pretrained')
        model_info_exists = path.isfile(path.join(dir_path, 'model_info.dat'))
        model_exists = path.isfile(path.join(dir_path, 'model.h5'))
        if (not model_exists or not model_info_exists):
            print('The pre-trained models to be downloaded for the intent extraction dataset '
                  'are licensed under Apache 2.0. By downloading, you accept the terms '
                  'and conditions provided by the license')
            makedirs(dir_path, exist_ok=True)
            if prompt is True:
                agreed = IntentExtractionApi._prompt()
                if agreed is False:
                    sys.exit(0)
            download_unlicensed_file('http://nervana-modelzoo.s3.amazonaws.com/NLP/intent/',
                                     'model_info.dat', self.model_info_path)
            download_unlicensed_file('http://nervana-modelzoo.s3.amazonaws.com/NLP/intent/',
                                     'model.h5', self.model_path)
            print('Done.')

    def display_results(self, text_str, predictions, intent_type):
        ret = {'annotation_set': []}
        ret['doc_text'] = ' '.join([t for t in text_str])
        counter = 0
        spans = []
        for t, n in zip(text_str, predictions):
            if n != 'O':
                ret['annotation_set'].append(n.lower())
                spans.append({
                    'start': counter,
                    'end': counter + len(t),
                    'type': n.lower()
                })
            counter += len(t) + 1
        ret['spans'] = spans
        ret['title'] = intent_type
        return {"doc": ret, 'type': 'high_level'}

    def vectorize(self, doc, vocab, char_vocab=None):
        words = np.asarray([vocab[w.lower()] if w.lower() in vocab else 1 for w in doc])\
            .reshape(1, -1)
        if char_vocab is not None:
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
            return [words, sentence_chars]
        else:
            return words

    def inference(self, doc):
        text_arr = self.process_text(doc)
        intent_type = None
        if self.model_type == 'mtl':
            doc_vec = self.vectorize(text_arr, self.word_vocab, self.char_vocab)
            intent, tags = self.model.predict(doc_vec, batch_size=1)
            intent = int(intent.argmax(1).flatten())
            intent_type = self.intent_vocab.get(intent, None)
            print('Detected intent type: {}'.format(intent_type))
        else:
            doc_vec = self.vectorize(text_arr, self.word_vocab, None)
            tags = self.model.predict(doc_vec, batch_size=1)
        tags = tags.argmax(2).flatten()
        tag_str = [self.tags_vocab.get(n, None) for n in tags]
        for t, n in zip(text_arr, tag_str):
            print('{}\t{}\t'.format(t, n))
        return self.display_results(text_arr, tag_str, intent_type)

    def load_model(self):
        if self.model_type == 'seq2seq':
            model = Seq2SeqIntentModel()
        else:
            model = MultiTaskIntentModel()
        model.load(self.pretrained_model)
        self.model = model
