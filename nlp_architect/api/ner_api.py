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

from os import path, makedirs, sys
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from nlp_architect.api.abstract_api import AbstractApi
from nlp_architect.utils.io import download_unlicensed_file
from nlp_architect.models.ner_crf import NERCRF

# from nlp_architect.utils.text import SpacyInstance

#nlp = SpacyInstance(disable=['tagger', 'ner', 'parser', 'vectors', 'textcat'])


class NerApi(AbstractApi):
    """
    Ner model API
    """
    dir = path.dirname(path.realpath(__file__))
    pretrained_model = path.join(dir, 'ner-pretrained', 'model.h5')
    pretrained_model_info = path.join(dir, 'ner-pretrained', 'model_info.dat')

    def __init__(self, ner_model=None, prompt=True):
        self.model = None
        self.model_info = None
        self.model_path = NerApi.pretrained_model
        self.model_info_path = NerApi.pretrained_model_info
        self._download_pretrained_model(prompt)

    def encode_word(self, word):
        return self.model_info['word_vocab'].get(word, 1.0)

    def encode_word_chars(self, word):
        return [self.model_info['char_vocab'].get(c, 1.0) for c in word]

    def encode_input(self, text_arr):
        print (text_arr)
        sentence = []
        for word in text_arr:
            sentence.append(self.encode_word(word))
        encoded_sentence = pad_sequences(
            [np.asarray(sentence)], maxlen=self.model_info['sentence_len'])
        return encoded_sentence


    def _prompt(self):
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
        dir_path = path.join(self.dir, 'ner-pretrained')
        if not path.isfile(path.join(dir_path, 'model.h5')):
            print('The pre-trained models to be downloaded for the NER dataset '
                  'are licensed under Apache 2.0. By downloading, you accept the terms '
                  'and conditions provided by the license')
            makedirs(dir_path, exist_ok=True)
            if prompt is True:
                agreed = self._prompt()
                if agreed is False:
                    sys.exit(0)
            download_unlicensed_file('http://nervana-modelzoo.s3.amazonaws.com/NLP/NER/',
                                     'model.h5', self.model_path)
            download_unlicensed_file('http://nervana-modelzoo.s3.amazonaws.com/NLP/NER/',
                                     'model_info.dat', self.model_info_path)
            print('Done.')

    def load_model(self):
        with open(self.model_info_path, 'rb') as fp:
            self.model_info = pickle.load(fp)
            self.model = NERCRF()
            self.model.build(
                self.model_info['sentence_len'],
                self.model_info['word_len'],
                self.model_info['num_of_labels'],
                self.model_info['word_vocab'],
                self.model_info['vocab_size'],
                self.model_info['char_vocab_size'],
                word_embedding_dims=self.model_info['word_embedding_dims'],
                char_embedding_dims=self.model_info['char_embedding_dims'],
                word_lstm_dims=self.model_info['word_lstm_dims'],
                tagger_lstm_dims=self.model_info['tagger_lstm_dims'],
                dropout=self.model_info['dropout'],
                external_embedding_model=self.model_info[
                    'external_embedding_model'])
            self.model.load(self.model_path)

    def pretty_print(self, text, tags):
        tags_str = [self.model_info['labels_id_to_word']
                    .get(t, None) for t in tags[0]][-len(text):]
        mapped = [
            {'index': idx, 'word': el, 'label': tags_str[idx]} for idx, el in enumerate(text)
        ]
        counter = 0
        ents = []
        spans = []
        words = []
        word = ''
        i = 0
        while i < len(mapped):
            word = word + mapped[i]['word']
            if mapped[i]['label'].startswith('B-'):
                i = i + 1
                while not mapped[i]['label'].startswith('E-'):
                    word = word + mapped[i]['word']
                    i = i + 1
                if(mapped[i]['label'].startswith('E-')):
                    word = word + mapped[i]['word']
                    spans.append({
                        'start': counter,
                        'end': (counter + len(word)),
                        'type': mapped[i]['label'].replace('E-','')
                    })
            words.append(word)
            counter += len(word) + 1
            word = ''
            i = i + 1
            
        ents = dict((obj['type'].lower(), obj) for obj in spans).keys()
        ret = {}
        ret['doc_text'] = ' '.join(words)
        ret['annotation_set'] = list(ents)
        ret['spans'] = spans
        ret['title'] = 'None'
        return {"doc": ret, 'type': 'high_level'}

    def process_text(self, text):
        input_text = ' '.join(text.strip().split())
        return [text_arr for text_arr in input_text]
        

    def inference(self, doc):
        text_arr = self.process_text(doc)
        words = self.encode_input(text_arr)
        tags = self.model.predict(words)
        tags = tags.argmax(2)
        print (tags)
        return self.pretty_print(text_arr, tags)
