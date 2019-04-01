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
import json
from enum import Enum
from json import JSONEncoder


class LexiconElement(object):
    def __init__(self, term: list, score: str or float = None, polarity: str = None,
                 is_acquired: str = None, position: str = None):
        self.term = term
        self.polarity = polarity
        try:
            self.score = float(score)
        except TypeError:
            self.score = 0
        self.position = position
        if is_acquired == "N":
            self.is_acquired = False
        elif is_acquired == "Y":
            self.is_acquired = True
        else:
            self.is_acquired = None

    def __lt__(self, other):
        return self.term[0] < other.term[0]

    def __le__(self, other):
        return self.term[0] <= other.term[0]

    def __eq__(self, other):
        return self.term[0] == other.term[0]

    def __ne__(self, other):
        return self.term[0] != other.term[0]

    def __gt__(self, other):
        return self.term[0] > other.term[0]

    def __ge__(self, other):
        return self.term[0] >= other.term[0]


class TermType(Enum):
    OPINION = 'OP'
    ASPECT = 'AS'
    NEGATION = 'NEG'
    INTENSIFIER = 'INT'


class Polarity(Enum):
    POS = 'POS'
    NEG = 'NEG'
    UNK = 'UNK'


class Term(object):
    def __init__(self, text: str, kind: TermType, polarity: Polarity, score: float, start: int,
                 length: int):
        self._text = text
        self._type = kind
        self._polarity = polarity
        self._score = score
        self._start = start
        self._len = length

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def text(self):
        return self._text

    @property
    def type(self):
        return self._type

    @property
    def polarity(self):
        return self._polarity

    @property
    def score(self):
        return self._score

    @property
    def start(self):
        return self._start

    @property
    def len(self):
        return self._len

    @text.setter
    def text(self, val):
        self._text = val

    @score.setter
    def score(self, val):
        self._score = val

    @polarity.setter
    def polarity(self, val):
        self._polarity = val

    def __str__(self):
        return "text: " + self._text + " type: " + str(self._type) + " pol: " + \
               str(self._polarity) + " score: " + str(self._score) + " start: " + \
               str(self._start) + " len: " + \
               str(self._len)


class SentimentDoc(object):
    def __init__(self, doc_text: str = None, sentences: list = None):
        if sentences is None:
            sentences = []
        self._doc_text = doc_text
        self._sentences = sentences

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def doc_text(self):
        return self._doc_text

    @doc_text.setter
    def doc_text(self, val):
        self._doc_text = val

    @property
    def sentences(self):
        return self._sentences

    @sentences.setter
    def sentences(self, val):
        self._sentences = val

    @staticmethod
    def decoder(obj):
        """
        :param obj: object to be decoded
        :return: decoded Sentence object
        """
        # SentimentDoc
        if '_doc_text' in obj and '_sentences' in obj:
            return SentimentDoc(obj['_doc_text'], obj['_sentences'])

        # SentimentSentence
        if all((attr in obj for attr in ('_start', '_end', '_events'))):
            return SentimentSentence(obj['_start'], obj['_end'], obj['_events'])

        # Term
        if all(attr in obj for attr in
               ('_text', '_type', '_score', '_polarity', '_start', '_len')):
            return Term(obj['_text'], TermType[obj['_type']],
                        Polarity[obj['_polarity']], obj['_score'], obj['_start'],
                        obj['_len'])
        return obj

    def __repr__(self):
        return self.pretty_json()

    def __str__(self):
        return self.__repr__()

    def __iter__(self):
        return self.sentences.__iter__()

    def __len__(self):
        return len(self.sentences)

    def json(self):
        """
        Return json representations of the object

        Returns:
            :obj:`json`: json representations of the object
        """
        return json.dumps(self, cls=SentimentDocEncoder)

    def pretty_json(self):
        """
        Return pretty json representations of the object

        Returns:
            :obj:`json`: pretty json representations of the object
        """
        return json.dumps(self, cls=SentimentDocEncoder, indent=4)


class SentimentSentence(object):
    def __init__(self, start: int, end: int, events: list):
        self._start = start
        self._end = end
        self._events = events

    def __eq__(self, other):
        if type(other) is type(self):
            return self.__dict__ == other.__dict__
        return False

    @property
    def start(self):
        return self._start

    @start.setter
    def start(self, val):
        self._start = val

    @property
    def end(self):
        return self._end

    @end.setter
    def end(self, val):
        self._end = val

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, val):
        self._events = val


class SentimentDocEncoder(JSONEncoder):
    def default(self, o):  # pylint: disable=E0202
        try:
            if isinstance(o, Enum):
                return getattr(o, '_name_')
            if hasattr(o, '__dict__'):
                return vars(o)
            if hasattr(o, '__slots__'):
                ret = {slot: getattr(o, slot) for slot in o.__slots__}
                for cls in type(o).mro():
                    spr = super(cls, o)
                    if not hasattr(spr, '__slots__'):
                        break
                    for slot in spr.__slots__:
                        ret[slot] = getattr(o, slot)
                return ret
        except Exception as e:
            print(e)
        return None
