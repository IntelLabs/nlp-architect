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
import csv
from enum import Enum
from os import PathLike

from nlp_architect.models.absa import TRAIN_LEXICONS


class OpinionTerm:
    """Opinion term.

    Attributes:
       terms (list): list of opinion term
        polarity (Polarity): polarity of the sentiment
    """

    def __init__(self, terms, polarity):
        self.terms = terms
        self.polarity = polarity

    def __str__(self):
        return ' '.join(self.terms)


class AspectTerm(object):
    """Aspect term.

    Attributes:
        terms (list): list of terms
        pos (list): list of pos
    """

    def __init__(self, terms, pos):
        """
        Args:
            terms (list): list of terms
            pos (list): list of pos
        """
        self.terms = terms
        self.pos = pos

    def __str__(self):
        return ' '.join(self.terms)

    def __eq__(self, other):
        """
        Override the default equals behavior.
        """
        return self.terms == other.terms and self.pos == other.pos

    @staticmethod
    def from_token(token):
        return AspectTerm([token.text], [token.norm_pos])


class CandidateTerm(object):
    """Candidate opinion term or aspect term.

    Attributes:
        term (list): list of terms
        pos (list): list of pos
        source_term (list): list of related anchor terms
        sentence (str): sentence text this term
        term_polarity (int): term polarity
    """

    def __init__(self, term_a, term_b, sent_text, candidate_term_polarity):
        """
        Args:
            term_a (DepRelationTerm): first term
            term_b (DepRelationTerm): second term
            sent_text (str): sentence text
            candidate_term_polarity (Polarity): term polarity
        """
        self.term = [term_a.text]
        self.pos = [term_a.norm_pos]
        self.source_term = [term_b.text]
        self.sentence = sent_text
        self.term_polarity = candidate_term_polarity

    def __str__(self):
        return ' '.join(self.term)

    def __eq__(self, other):
        if other is None or self.__class__ != other.__class__:
            return False
        if self.term != other.term if self.term is not None else other.term is not None:
            return False
        if self.source_term != other.source_term if self.source_term is not None else \
                other.source_term is not None:
            return False
        return self.sentence == other.sentence if self.sentence is not None else \
            other.sentence is None

    def __ne__(self, other):
        return not self == other


class DepRelation(object):
    """Generic Relation Entry contains the governor, it's dependent and the relation between them.

    Attributes:
        gov (DepRelationTerm): governor
        dep (DepRelationTerm): dependent
        rel (str): relation type between governor and dependent
    """

    def __init__(self, gov=None, dep=None, rel=None):
        self.gov = gov
        self.dep = dep
        rel_split = rel.split(':')
        self.rel = rel_split[0]
        self.subtype = rel_split[1] if len(rel_split) > 1 else None


class RelCategory(Enum):
    SUBJ = {'nsubj', 'nsubjpass', 'csubj', 'csubjpass'}
    MOD = {'amod', 'acl', 'advcl', 'appos', 'neg', 'nmod'}
    OBJ = {'dobj', 'iobj'}


class DepRelationTerm(object):
    """
    Attributes:
        text (str, optional): token text
        lemma (str, optional): token lemma
        pos (str, optional): token pos
        ner (str, optional): token ner
        idx (int, optional): token start index (within the sentence)
    """

    def __init__(self, text=None, lemma=None, pos=None, ner=None, idx=None):
        self.text = text
        self.lemma = lemma
        self.pos = pos
        self.ner = ner
        self.idx = idx
        self.dep_rel_list = []
        self.gov = None

    @property
    def norm_pos(self):
        return normalize_pos(self.text, self.pos)


def string_list_headers():
    return ["CandidateTerm", "Frequency", "Polarity"]


class QualifiedTerm(object):
    """Qualified term - term that is accepted to generated lexicon.

    Attributes:
        term (list): list of terms
        pos (list): list of pos.
        frequency (int): frequency of filtered term in corpus.
        term_polarity (Polarity): term polarity.

    """

    def __init__(self, term, pos, frequency, term_polarity):
        self.term = term
        self.pos = pos
        self.frequency = frequency
        self.term_polarity = term_polarity

    def as_string_list(self):
        return [' '.join(self.term), str(self.frequency),
                self.term_polarity.name]


def load_lex_as_dict_from_csv(file_name: str or PathLike):
    """Read lexicon as dictionary, key = term, value = pos.

    Args:
        file_name: the csv file name
    """
    lexicon_map = {}
    with open(file_name) as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        if reader is None:
            print("file name is None")
            return lexicon_map
        next(reader)
        for row in reader:
            term = row['Term']
            pos = row['POS subtype']

            lexicon_map[term] = pos
    return lexicon_map


class POS(Enum):
    """Part-of-speech labels."""
    ADJ = 1
    ADV = 2
    AUX = 3
    AUX_PAST = 3
    CONJ = 4
    NUM = 5
    DET = 6
    EX = 7
    FW = 8
    IN = 9
    PREP = 10
    LS = 11
    MD = 12
    MD_CERTAIN = 13
    NN = 14
    PROPER_NAME = 15
    POS = 16
    PRON = 17
    PRON_1_S = 18
    PRON_1_P = 19
    PRON_2_S = 20
    PRON_3_S = 21
    PRON_3_P = 22
    PRON_4_S = 23
    POSSPRON_1_S = 24
    POSSPRON_1_P = 25
    POSSPRON_2_S = 26
    POSSPRON_2_P = 27
    POSSPRON_3_S = 28
    POSSPRON_3_P = 29
    POSSPRON_4_S = 30
    POSSPRON_4_P = 31
    RP = 32
    SYM = 33
    TO = 34
    INTERJ = 35
    VB = 36
    VB_PAST = 37
    VB_PRESENT = 38
    VBG = 39
    VBN = 40
    WH_DET = 41
    WH_PROP = 42
    WH_ADV = 43
    PUNCT = 44
    OTHER = 45


PRONOUNS_LIST = load_lex_as_dict_from_csv(TRAIN_LEXICONS / 'PronounsLex.csv')


def normalize_pos(word, in_pos):
    if in_pos is None:
        return POS.OTHER
    if word.lower() in PRONOUNS_LIST and in_pos.startswith("PR"):
        return POS[PRONOUNS_LIST[word.lower()]]
    if in_pos == "CC":
        return POS.CONJ
    if in_pos == "CD":
        return POS.NUM
    if in_pos == "DT":
        return POS.DET
    if in_pos == "EX":
        return POS.EX
    if in_pos == "FW":
        return POS.FW
    if in_pos == "IN":
        return POS.PREP
    if in_pos == "TO":
        return POS.PREP
    if in_pos.startswith("JJ"):
        return POS.ADJ
    if in_pos == "LS":
        return POS.LS
    if in_pos == "MD":
        return POS.MD
    if in_pos.startswith("NN"):
        return POS.NN
    if in_pos == "PDT":
        return POS.DET
    if in_pos == "POS":
        return POS.POS
    if in_pos.startswith("PR"):
        return POS.PRON
    if in_pos.startswith("RB"):
        return POS.ADV
    if in_pos == "RP":
        return POS.RP
    if in_pos == "SYM":
        return POS.SYM
    if in_pos == "UH":
        return POS.INTERJ
    if in_pos.startswith("VB"):
        return POS.VB
    if in_pos == "WDT":
        return POS.WH_DET
    if in_pos.startswith("WP"):
        return POS.WH_PROP
    if in_pos == "WRB":
        return POS.WH_ADV
    return POS.OTHER


class LoadAspectStopLists(object):
    """A Filter holding all generic and general lexicons, can verify if a given term is contained
     in one of the lexicons - hence belongs to one of the generic / general lexicons or is a valid
     term.

    Attributes:
        generic_opinion_lex (dict): generic opinion lexicon
        determiners_lex (dict): determiners lexicon
        general_adjectives_lex (dict): general adjectives lexicon
        generic_quantifiers_lex (dict): generic quantifiers lexicon
        geographical_adjectives_lex (dict): geographical adjectives lexicon
        intensifiers_lex (dict): intensifiers lexicon
        time_adjective_lex (dict): time adjective lexicon
        ordinal_numbers_lex (dict): ordinal numbers lexicon
        prepositions_lex (dict): prepositions lexicon
        pronouns_lex (dict): pronouns lexicon
        colors_lex (dict): colors lexicon
        negation_lex (dict): negation terms lexicon
    """

    def __init__(self, generic_opinion_lex, determiners_lex, general_adjectives_lex,
                 generic_quantifiers_lex, geographical_adjectives_lex, intensifiers_lex,
                 time_adjective_lex, ordinal_numbers_lex, prepositions_lex, pronouns_lex,
                 colors_lex, negation_lex):
        self.generic_opinion_lex = generic_opinion_lex
        self.determiners_lex = determiners_lex
        self.general_adjectives_lex = general_adjectives_lex
        self.generic_quantifiers_lex = generic_quantifiers_lex
        self.geographical_adjectives_lex = geographical_adjectives_lex
        self.intensifiers_lex = intensifiers_lex
        self.time_adjective_lex = time_adjective_lex
        self.ordinal_numbers_lex = ordinal_numbers_lex
        self.prepositions_lex = prepositions_lex
        self.pronouns_lex = pronouns_lex
        self.colors_lex = colors_lex
        self.negation_lex = negation_lex

    def is_in_stop_list(self, term):
        return any(term in lexicon for lexicon in self.__dict__.values())


class LoadOpinionStopLists(object):
    """A Filter holding all generic and general lexicons, can verify if a given term is contained
     in one of the lexicons - hence belongs to one of the generic / general lexicons or is a valid
     term.

    Attributes:
        determiners_lex (dict): determiners lexicon
        general_adjectives_lex (dict): general adjectives lexicon
        generic_quantifiers_lex (dict): generic quantifiers lexicon
        geographical_adjectives_lex (dict): geographical adjectives lexicon
        intensifiers_lex (dict): intensifiers lexicon
        time_adjective_lex (dict): time adjective lexicon
        ordinal_numbers_lex (dict): ordinal numbers lexicon
        prepositions_lex (dict): prepositions lexicon
        colors_lex (dict): colors lexicon
        negation_lex (dict): negation terms lexicon
    """

    def __init__(self, determiners_lex, general_adjectives_lex, generic_quantifiers_lex,
                 geographical_adjectives_lex, intensifiers_lex, time_adjective_lex,
                 ordinal_numbers_lex, prepositions_lex, colors_lex, negation_lex):
        self.determiners_lex = determiners_lex
        self.general_adjectives_lex = general_adjectives_lex
        self.generic_quantifiers_lex = generic_quantifiers_lex
        self.geographical_adjectives_lex = geographical_adjectives_lex
        self.intensifiers_lex = intensifiers_lex
        self.time_adjective_lex = time_adjective_lex
        self.ordinal_numbers_lex = ordinal_numbers_lex
        self.prepositions_lex = prepositions_lex
        self.colors_lex = colors_lex
        self.negation_lex = negation_lex

    def is_in_stop_list(self, term):
        return any(term in lexicon for lexicon in self.__dict__.values())
