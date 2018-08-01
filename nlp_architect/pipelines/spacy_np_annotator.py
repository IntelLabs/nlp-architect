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

import numpy as np
from os import path
import spacy
from spacy.tokens import Doc
from spacy.tokens import Span

from nlp_architect.models.chunker import SequenceChunker
from nlp_architect.utils.generic import pad_sentences
from nlp_architect.utils.io import validate_existing_filepath
from nlp_architect.utils.text import extract_nps


class NPAnnotator(object):
    """
    Spacy based NP annotator - uses models.SequenceChunker model for annotation

    Args:
        model (SequenceChunker): a chunker model
        word_vocab (Vocabulary): word-id vocabulary of the model
        chunk_vocab (Vocabulary): chunk tag vocabulary of the model
        batch_size (int, optional): inference batch size
    """

    def __init__(self, model, word_vocab, chunk_vocab, batch_size: int = 32):
        self.model = model
        self.bs = batch_size
        self.word_vocab = word_vocab
        self.chunk_vocab = chunk_vocab
        Doc.set_extension('noun_phrases', default=[])

    @classmethod
    def load(cls, model_path: str, parameter_path: str, batch_size: int = 32,
             use_gpu: bool = False):
        """
        Load a NPAnnotator annotator

        Args:
            model_path (str): path to trained model
            parameter_path (str): path to model parameters
            batch_size (int, optional): inference batch_size
            use_gpu (bool, optional): use gpu for inference (cudnn cells)

        Returns:
            NPAnnotator class with loaded model
        """

        _model_path = path.join(path.dirname(path.realpath(__file__)), model_path)
        validate_existing_filepath(_model_path)
        _parameter_path = path.join(path.dirname(path.realpath(__file__)), parameter_path)
        validate_existing_filepath(_parameter_path)

        model = SequenceChunker(use_gpu=use_gpu)
        model.load(_model_path)
        with open(_parameter_path, 'rb') as fp:
            model_params = pickle.load(fp)
            word_vocab = model_params['word_vocab']
            chunk_vocab = model_params['chunk_vocab']
        model.chunk_inference_mode()
        return cls(model, word_vocab, chunk_vocab, batch_size)

    def _infer_chunks(self, vec, doc_lengths):
        tagged_sents = self.model.predict(vec, batch_size=self.bs).argmax(2)
        sentence = []
        for c, l in zip(tagged_sents, doc_lengths):
            sentence.append(c[-l:])
        doc = np.concatenate(sentence)
        chunk_tags = [self.chunk_vocab.id_to_word(w) for w in doc]
        return extract_nps(chunk_tags)

    def _feature_extractor(self, doc):
        return np.asarray([self.word_vocab[w] if self.word_vocab[w] is not None else 1
                           for w in doc])

    def __call__(self, doc: Doc) -> Doc:
        """
        Annotate the document with noun phrase spans
        """
        spans = []
        doc_vecs = []
        doc_lens = []
        for sentence in doc.sents:
            doc_vec = self._feature_extractor([t.text for t in sentence])
            doc_vecs.append(doc_vec)
            doc_lens.append(len(doc_vec))
        doc_vectors = pad_sentences(doc_vecs)
        np_indexes = self._infer_chunks(doc_vectors, doc_lens)
        for s, e in np_indexes:
            np_span = Span(doc, s, e)
            spans.append(np_span)
        spans = _NPPostprocessor.process(spans)
        set_noun_phrases(doc, spans)
        return doc


def get_noun_phrases(doc: Doc) -> [Span]:
    """
    Get noun phrase tags from a spacy annotated document.

    Args:
        doc (Doc): a spacy type document

    Returns:
        a list of noun phrase Span objects
    """
    assert hasattr(doc._, 'noun_phrases'), 'no noun_phrase attributes in document'
    return doc._.noun_phrases


def set_noun_phrases(doc: Doc, nps: [Span]) -> None:
    """
    Set noun phrase tags

    Args:
        doc (Doc): a spacy type document
        nps ([Span]): a list of Spans
    """
    assert hasattr(doc._, 'noun_phrases'), 'no noun_phrase attributes in document'
    doc._.set('noun_phrases', nps)


class _NPPostprocessor:
    @classmethod
    def process(cls, noun_phrases: [Span]) -> [Span]:
        new_phrases = []
        for phrase in noun_phrases:
            p = _NPPostprocessor._phrase_process(phrase)
            if p is not None and len(p) > 0:
                new_phrases.append(p)
        return new_phrases

    @classmethod
    def _phrase_process(cls, phrase: Span) -> Span:
        last_phrase = None
        while phrase != last_phrase:
            for func_args in post_processing_rules:
                pf = func_args[0]
                args = func_args[1:]
                if len(args) > 0:
                    phrase = pf(phrase, *args)
                else:
                    phrase = pf(phrase)
                if phrase is None:
                    break
            last_phrase = phrase
        return phrase


def _filter_repeating_nonalnum(phrase, length):
    """
    Check if a given phrase has non repeating alphanumeric chars
    of given length.
    Example: 'phrase $$$' with length=3 will return False
    """
    if len(phrase) > 0:
        alnum_len = length
        for t in phrase:
            if not t.is_alpha:
                alnum_len -= 1
            else:
                alnum_len = length
            if alnum_len == 0:
                return None
    return phrase


def _filter_long_phrases(phrase, word_length, phrase_length):
    if len(phrase) > 0 and max([len(t) for t in phrase]) > word_length \
            and len(phrase) > phrase_length:
        return None
    return phrase


def _remove_non_alphanum_from_start(phrase):
    if len(phrase) > 1 and not phrase[0].is_alpha:
        phrase = phrase[1:]
    return phrase


def _remove_non_alphanum_from_end(phrase):
    if len(phrase) > 1 and not phrase[-1].is_alpha:
        phrase = phrase[:-1]
    return phrase


def _remove_stop_words(phrase):
    while len(phrase) > 0 and phrase[0].is_stop:
        phrase = phrase[1:]
    while len(phrase) > 0 and phrase[-1].is_stop:
        phrase = phrase[:-1]
    return phrase


def _remove_char_at_start(phrase):
    chars = ['@', '-', '=', '.', ':', '+', '?', 'nt', '\"', '\'', '\'S', '\'s', ',']
    if phrase and len(phrase) > 0:
        while len(phrase) > 0 and phrase[0].text in chars:
            phrase = phrase[1:]
    return phrase


def _remove_char_at_end(phrase):
    chars = [',', '(', ')', ' ']
    if phrase:
        while len(phrase) > 0 and phrase[-1].text in chars:
            phrase = phrase[:-1]
    return phrase


def _remove_pos_from_start(phrase):
    tag_list = ['WDT', 'PRP$']
    pos_list = ['PUNCT', 'INTJ', 'NUM', 'PART', 'ADV', 'DET', 'PRON', 'VERB']
    if phrase:
        while len(phrase) > 0 and (phrase[0].pos_ in pos_list or phrase[0].tag_ in tag_list):
            phrase = phrase[1:]
    return phrase


def _remove_pos_from_end(phrase):
    tag_list = ['WDT']
    pos_list = ['DET', 'PUNCT', 'CONJ']
    if phrase:
        while len(phrase) > 0 and (phrase[-1].pos_ in pos_list or phrase[-1].tag_ in tag_list):
            phrase = phrase[:-1]
    return phrase


def _filter_single_pos(phrase):
    pos_list = ['VERB', 'ADJ', 'ADV']
    if phrase and len(phrase) == 1 and phrase[0].pos_ in pos_list:
        return None
    return phrase


def _filter_fp_nums(phrase):
    if len(phrase) > 0:
        try:
            # check for float number
            float(phrase.text.replace(',', ''))
            return None
        except ValueError:
            return phrase
    return phrase


def _filter_single_char(phrase):
    if phrase and len(phrase) == 1 and len(phrase[0]) == 1:
        return None
    return phrase


def _filter_empty(phrase):
    if phrase is None or len(phrase) == 0 or len(phrase.text) == 0:
        return None
    return phrase


post_processing_rules = [
    (_filter_single_char,),
    (_filter_single_pos,),
    (_remove_pos_from_start,),
    (_remove_pos_from_end,),
    (_remove_stop_words,),
    (_remove_non_alphanum_from_start,),
    (_remove_non_alphanum_from_end,),
    (_filter_repeating_nonalnum, 5),
    (_filter_long_phrases, 5, 75),
    (_remove_char_at_start,),
    (_remove_char_at_end,),
    (_filter_fp_nums,),
    (_filter_empty,),
]


class SpacyNPAnnotator(object):
    """
    Simple Spacy pipe with NP extraction annotations
    """

    def __init__(self, model_path, settings_path, spacy_model='en', batch_size=32, use_gpu=False):
        _model_path = path.join(path.dirname(path.realpath(__file__)), model_path)
        validate_existing_filepath(_model_path)
        _settings_path = path.join(path.dirname(path.realpath(__file__)), settings_path)
        validate_existing_filepath(_settings_path)

        nlp = spacy.load(spacy_model)
        for p in nlp.pipe_names:
            if p not in ['tagger']:
                nlp.remove_pipe(p)
        nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
        nlp.add_pipe(NPAnnotator.load(_model_path, settings_path, batch_size=batch_size,
                                      use_gpu=use_gpu), last=True)
        self.nlp = nlp

    def __call__(self, text: str) -> [str]:
        """
        Parse a given text and return a list of noun phrases found

        Args:
            text (str): a text string

        Returns:
            list of noun phrases as strings
        """
        return [np.text for np in get_noun_phrases(self.nlp(text))]
