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
from nlp_architect import LIBRARY_OUT
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import download_unlicensed_file
from nlp_architect.utils.text import SpacyInstance, bio_to_spans


class IntentExtractionApi(AbstractApi):
    model_dir = str(LIBRARY_OUT / 'intent-pretrained')
    pretrained_model_info = path.join(model_dir, 'model_info.dat')
    pretrained_model = path.join(model_dir, 'model.h5')

    def __init__(self, prompt=True):
        self.model = None
        self.model_type = None
        self.word_vocab = None
        self.tags_vocab = None
        self.char_vocab = None
        self.intent_vocab = None
        self._download_pretrained_model(prompt)
        self.nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])

    def process_text(self, text):
        input_text = ' '.join(text.strip().split())
        return self.nlp.tokenize(input_text)

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

    @staticmethod
    def _download_pretrained_model(prompt=True):
        """Downloads the pre-trained BIST model if non-existent."""
        model_info_exists = path.isfile(IntentExtractionApi.pretrained_model_info)
        model_exists = path.isfile(IntentExtractionApi.pretrained_model)
        if not model_exists or not model_info_exists:
            print('The pre-trained models to be downloaded for the intent extraction dataset '
                  'are licensed under Apache 2.0. By downloading, you accept the terms '
                  'and conditions provided by the license')
            makedirs(IntentExtractionApi.model_dir, exist_ok=True)
            if prompt is True:
                agreed = IntentExtractionApi._prompt()
                if agreed is False:
                    sys.exit(0)
            download_unlicensed_file('https://s3-us-west-2.amazonaws.com/nlp-architect-data'
                                     '/models/intent/',
                                     'model_info.dat', IntentExtractionApi.pretrained_model_info)
            download_unlicensed_file('https://s3-us-west-2.amazonaws.com/nlp-architect-data'
                                     '/models/intent/',
                                     'model.h5', IntentExtractionApi.pretrained_model)
            print('Done.')

    @staticmethod
    def display_results(text_str, predictions, intent_type):
        ret = {'annotation_set': [], 'doc_text': ' '.join([t for t in text_str])}
        spans = []
        available_tags = set()
        for s, e, tag in bio_to_spans(text_str, predictions):
            spans.append({
                'start': s,
                'end': e,
                'type': tag
            })
            available_tags.add(tag)
        ret['annotation_set'] = list(available_tags)
        ret['spans'] = spans
        ret['title'] = intent_type
        return {'doc': ret, 'type': 'high_level'}

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
        with open(IntentExtractionApi.pretrained_model_info, 'rb') as fp:
            model_info = pickle.load(fp)
        self.model_type = model_info['type']
        self.word_vocab = model_info['word_vocab']
        self.tags_vocab = {v: k for k, v in model_info['tags_vocab'].items()}
        if self.model_type == 'mtl':
            self.char_vocab = model_info['char_vocab']
            self.intent_vocab = {v: k for k, v in model_info['intent_vocab'].items()}
            model = MultiTaskIntentModel()
        else:
            model = Seq2SeqIntentModel()
        model.load(self.pretrained_model)
        self.model = model
