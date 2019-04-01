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
import copy
import re
import sys
from os import PathLike

from tqdm import tqdm

from nlp_architect.models.absa import TRAIN_OUT
from nlp_architect.models.absa.inference.data_types import Polarity
from nlp_architect.models.absa.train.data_types import AspectTerm, \
    DepRelation, DepRelationTerm, LoadOpinionStopLists, LoadAspectStopLists, OpinionTerm, \
    QualifiedTerm
from nlp_architect.models.absa import TRAIN_LEXICONS
from nlp_architect.models.absa.utils import _load_parsed_docs_from_dir, _write_final_lex, \
    _load_lex_as_list_from_csv, read_generic_lex_from_file
from nlp_architect.models.absa.train.rules import rule_1, rule_2, rule_3, rule_4, rule_5, rule_6


class AcquireTerms(object):
    """
    Lexicon acquisition. produce opinion lexicon and an aspect lexicon based
    on input dataset.

    Attributes:
        opinion_candidate_list_curr_iter (dict): candidate opinion terms in the current iteration
        opinion_candidate_list_prev_iter (dict): opinion candidates list of previous iteration
        opinion_candidate_list (dict): opinion terms learned across all iterations
        opinion_candidates_list_final (list): final opinion candidates list
        opinion_candidate_list_raw (dict): all instances of candidate opinion terms
                                           across all iterations
        aspect_candidate_list_curr_iter (dict): candidate terms in the current iteration
        aspects_candidate_list_prev_iter(list): Aspect candidates list of previous iteration
        aspect_candidate_list (list):  aspect terms learned across all iterations
        aspect_candidates_list_final (list): final aspect candidates list
        aspect_candidate_list_raw (dict): all instances of candidate aspect terms
                                          across all iterations
        """

    out_dir = TRAIN_OUT / 'output'
    feature_table_path = out_dir / 'feature_table.csv'
    generic_opinion_lex_path = TRAIN_LEXICONS / 'GenericOpinionLex.csv'
    acquired_opinion_terms_path = out_dir / 'generated_opinion_lex.csv'
    acquired_aspect_terms_path = out_dir / 'generated_aspect_lex.csv'

    GENERIC_OPINION_LEX = _load_lex_as_list_from_csv('GenericOpinionLex.csv')
    GENERAL_ADJECTIVES_LEX = _load_lex_as_list_from_csv('GeneralAdjectivesLex.csv')
    GENERIC_QUANTIFIERS_LEX = _load_lex_as_list_from_csv('GenericQuantifiersLex.csv')
    GEOGRAPHICAL_ADJECTIVES_LEX = _load_lex_as_list_from_csv('GeographicalAdjectivesLex.csv')
    INTENSIFIERS_LEX = _load_lex_as_list_from_csv('IntensifiersLex.csv')
    TIME_ADJECTIVE_LEX = _load_lex_as_list_from_csv('TimeAdjectiveLex.csv')
    ORDINAL_NUMBERS_LEX = _load_lex_as_list_from_csv('OrdinalNumbersLex.csv')
    PREPOSITIONS_LEX = _load_lex_as_list_from_csv('PrepositionsLex.csv')
    PRONOUNS_LEX = _load_lex_as_list_from_csv('PronounsLex.csv')
    COLORS_LEX = _load_lex_as_list_from_csv('ColorsLex.csv')
    DETERMINERS_LEX = _load_lex_as_list_from_csv('DeterminersLex.csv')
    NEGATION_LEX = _load_lex_as_list_from_csv('NegationLex.csv')

    OPINION_STOP_LIST = LoadOpinionStopLists(DETERMINERS_LEX,
                                             GENERAL_ADJECTIVES_LEX,
                                             GENERIC_QUANTIFIERS_LEX,
                                             GEOGRAPHICAL_ADJECTIVES_LEX,
                                             INTENSIFIERS_LEX, TIME_ADJECTIVE_LEX,
                                             ORDINAL_NUMBERS_LEX,
                                             PREPOSITIONS_LEX, COLORS_LEX, NEGATION_LEX)

    ASPECT_STOP_LIST = LoadAspectStopLists(GENERIC_OPINION_LEX,
                                           DETERMINERS_LEX,
                                           GENERAL_ADJECTIVES_LEX,
                                           GENERIC_QUANTIFIERS_LEX,
                                           GEOGRAPHICAL_ADJECTIVES_LEX,
                                           INTENSIFIERS_LEX, TIME_ADJECTIVE_LEX,
                                           ORDINAL_NUMBERS_LEX,
                                           PREPOSITIONS_LEX, PRONOUNS_LEX, COLORS_LEX,
                                           NEGATION_LEX)

    FILTER_PATTERNS = [re.compile(r'.*\d+.*')]
    FLOAT_FORMAT = '{0:.3g}'
    # maximum number of iterations
    MAX_NUM_OF_ITERATIONS = 3
    NUM_OF_SENTENCES_PER_OPINION_AND_ASPECT_TERM_INC = 35000

    opinion_candidate_list_prev_iter = read_generic_lex_from_file(generic_opinion_lex_path)
    generic_sent_dict = copy.deepcopy(opinion_candidate_list_prev_iter)
    opinion_candidate_list = {}
    opinion_candidate_list_raw = {}
    opinion_candidate_list_curr_iter = {}
    opinion_candidates_list_final = []
    aspect_candidate_list_raw = {}
    aspect_candidate_list = list()
    aspect_candidate_list_curr_iter = {}
    aspect_candidates_list_final = []
    init_aspect_dict = list()
    aspects_candidate_list_prev_iter = list()
    min_freq_opinion_candidate = 2
    min_freq_aspect_candidate = 3

    def extract_terms_from_doc(self, parsed_doc):
        """Extract candidate terms for sentences in parsed document.

        Args:
            parsed_doc (ParsedDocument): Input parsed document.
        """
        for text, parsed_sent in parsed_doc.sent_iter():
            relations = _get_rel_list(parsed_sent)

            for rel_entry in relations:
                if rel_entry.rel != 'root':
                    gov_seen = self.opinion_candidate_list_prev_iter.get(rel_entry.gov.text)
                    dep_seen = self.opinion_candidate_list_prev_iter.get(rel_entry.dep.text)
                    opinions = []
                    aspects = []

                    # =========================== acquisition rules ==============================

                    if bool(gov_seen) ^ bool(dep_seen):
                        opinions.append(rule_1(rel_entry, gov_seen, dep_seen, text))

                    if not gov_seen and dep_seen:
                        opinions.append(rule_2(rel_entry, relations, dep_seen, text))

                        aspects.append(rule_3(rel_entry, relations, text))

                        aspects.append(rule_4(rel_entry, relations, text))

                    if self.aspects_candidate_list_prev_iter and \
                            AspectTerm.from_token(rel_entry.gov) \
                            in self.aspects_candidate_list_prev_iter and \
                            AspectTerm.from_token(rel_entry.dep) \
                            not in self.aspects_candidate_list_prev_iter:

                        opinions.append(rule_5(rel_entry, text))
                        aspects.append(rule_6(rel_entry, relations, text))

                    self._add_opinion_term(opinions)
                    self._add_aspect_term(aspects)

    def extract_opinion_and_aspect_terms(self, parsed_document_iter, num_of_docs):
        """Extract candidate terms from parsed document iterator.

        Args:
            parsed_document_iter (Iterator): Parsed document iterator.
            num_of_docs (int): number of documents on iterator.
        """

        for parsed_document in tqdm(parsed_document_iter, total=num_of_docs, file=sys.stdout):
            self.extract_terms_from_doc(parsed_document)

    def _is_valid_term(self, cand_term):
        """Validates a candidate term.

        Args:
            cand_term (CandidateTerm): candidate terms list.
        """
        term = str(cand_term)
        for pattern in self.FILTER_PATTERNS:
            if pattern.match(term):
                return False
        if self.OPINION_STOP_LIST.is_in_stop_list(term):
            return False
        if term.lower() != term and term.upper() != term:
            return False
        return True

    def _add_aspect_term(self, terms):
        """
        add new aspect term to table.
        Args:
            terms (list of CandidateTerm): candidate terms list
        """
        for term in terms:
            if term:
                term_entry = AspectTerm(term.term, term.pos)
                if term_entry not in self.init_aspect_dict and \
                        term_entry not in self.aspect_candidate_list and not\
                        self.ASPECT_STOP_LIST.is_in_stop_list(term.term[0]):
                    _insert_new_term_to_table(term, self.aspect_candidate_list_curr_iter)

        return True

    def _add_opinion_term(self, terms):
        """
        Add new opinion term to table
        Args:
            terms (list of CandidateTerm): candidate term
        """
        for term in terms:
            if term and self._is_valid_term(term):
                if str(term.term[0]) not in self.generic_sent_dict.keys():
                    if str(term.term[0]) not in self.opinion_candidate_list:
                        _insert_new_term_to_table(term, self.opinion_candidate_list_curr_iter)

    def _insert_new_terms_to_tables(self):
        """
        Insert new terms to tables
        clear candidates lists from previous iteration

        """
        self.opinion_candidate_list_prev_iter = {}
        self.opinion_candidate_list_raw = _merge_tables(self.opinion_candidate_list_raw,
                                                        self.opinion_candidate_list_curr_iter)
        for cand_term_list in self.opinion_candidate_list_curr_iter.values():
            if len(cand_term_list) >= \
                    self.min_freq_opinion_candidate:
                new_opinion_term = _set_opinion_term_polarity(cand_term_list)
                self.opinion_candidate_list_prev_iter[
                    str(new_opinion_term)] = new_opinion_term
        self.opinion_candidate_list_curr_iter = {}
        self.opinion_candidate_list = {**self.opinion_candidate_list,
                                       **self.opinion_candidate_list_prev_iter}
        self.aspects_candidate_list_prev_iter = list()
        self.aspect_candidate_list_raw = _merge_tables(
            self.aspect_candidate_list_raw,
            self.aspect_candidate_list_curr_iter)
        for extracted_aspect_list in self.aspect_candidate_list_curr_iter.values():
            if len(extracted_aspect_list) >= \
                    self.min_freq_aspect_candidate:
                first = extracted_aspect_list[0]
                new_aspect_entry = AspectTerm(first.term, first.pos)
                if new_aspect_entry not in self.aspects_candidate_list_prev_iter:
                    self.aspects_candidate_list_prev_iter.append(new_aspect_entry)
        self.aspect_candidate_list_curr_iter = {}
        self.aspect_candidate_list = \
            self.aspect_candidate_list + self.aspects_candidate_list_prev_iter

    def _write_output(self):
        """
        write generated lexicons to csv files
        """
        out = AcquireTerms.out_dir
        out.mkdir(parents=True, exist_ok=True)

        _write_final_lex(self.opinion_candidates_list_final, self.acquired_opinion_terms_path)
        _write_final_lex(self.aspect_candidates_list_final, self.acquired_aspect_terms_path)

    def acquire_lexicons(self, parsed_dir: str or PathLike):
        """Acquire new opinion and aspect lexicons.

        Args:
            parsed_dir (PathLike): Path to parsed documents folder.
        """
        parsed_docs = _load_parsed_docs_from_dir(parsed_dir)
        dataset_sentence_len = 0
        for parsed_doc in parsed_docs.values():
            dataset_sentence_len += len(parsed_doc.sentences)

        add_to_thresholds = \
            int(dataset_sentence_len / self.NUM_OF_SENTENCES_PER_OPINION_AND_ASPECT_TERM_INC)
        self.min_freq_opinion_candidate += add_to_thresholds
        self.min_freq_aspect_candidate += add_to_thresholds

        for iteration_num in range(self.MAX_NUM_OF_ITERATIONS):
            if len(self.opinion_candidate_list_prev_iter) == 0 \
                    and len(self.aspects_candidate_list_prev_iter) == 0:
                break

            print("\n#Iteration: {}".format(iteration_num + 1))

            self.extract_opinion_and_aspect_terms(iter(parsed_docs.values()),
                                                  len(parsed_docs))

            self._insert_new_terms_to_tables()

        self.opinion_candidates_list_final = \
            generate_final_opinion_candidates_list(
                self.opinion_candidate_list_raw, self.opinion_candidates_list_final,
                self.min_freq_opinion_candidate)
        self.aspect_candidates_list_final = \
            _generate_final_aspect_candidates_list(
                self.aspect_candidate_list_raw,
                self.aspect_candidates_list_final,
                self.min_freq_aspect_candidate)

        self._write_output()

        aspect_dict = {}
        for cand_term in self.aspect_candidates_list_final:
            aspect_dict[cand_term.term[0]] = cand_term.frequency

        return aspect_dict


def _get_rel_list(parsed_sentence):
    res = []
    gen_toks = []
    for tok in parsed_sentence:
        gen_toks.append(
            DepRelationTerm(tok['text'], tok['lemma'], tok['pos'], tok['ner'], tok['start']))

    for gen_tok, tok in zip(gen_toks, parsed_sentence):
        gov_idx = tok['gov']
        if gov_idx != -1:
            res.append(DepRelation(gen_toks[gov_idx], gen_tok, tok['rel']))
    return res


def _merge_tables(d1, d2):
    """
    Merge dictionaries
    Args:
        d1 (dict): first dict to merge
        d2 (dict): second dict to merge
    """
    for key, l in d2.items():
        if key in d1:
            for item in l:
                if item not in d1[key]:
                    d1[key].append(item)
        else:
            d1[key] = l
    return d1


def _insert_new_term_to_table(term, curr_table):
    """
    Insert term to table of lists.
    Args:
        term (term): term to be inserted
        curr_table (dict): input table
    """
    table_key_word = str(term)
    if table_key_word:
        if table_key_word in curr_table and term not in curr_table[table_key_word]:
            curr_table[table_key_word].append(term)
        else:
            curr_table[table_key_word] = [term]


def _set_opinion_term_polarity(terms_list):
    """Set opinion term polarity.

    Args:
        terms_list (list): list of opinion terms
    """
    first = terms_list[0]
    new_term = first.term

    positive_pol = 0
    negative_pol = 0
    pol = None
    for term in terms_list:
        try:
            pol = term.term_polarity
        except Exception as e:
            print("extracted_term missing term_polarity: " + str(e))
        if pol is not None:
            if pol == Polarity.POS:
                positive_pol = positive_pol + 1
            if pol == Polarity.NEG:
                negative_pol = negative_pol + 1
    new_term_polarity = Polarity.UNK
    if positive_pol >= negative_pol and positive_pol > 0:
        new_term_polarity = Polarity.POS
    elif negative_pol >= positive_pol and negative_pol > 0:
        new_term_polarity = Polarity.NEG

    return OpinionTerm(new_term, new_term_polarity)


def _generate_final_aspect_candidates_list(aspect_candidate_list_raw,
                                           final_aspect_candidates_list,
                                           frequency_threshold):
    """
    generate final aspect candidates list from map
    Args:
        aspect_candidate_list_raw (dict): key = term, value =
        lists of candidate terms.
        final_aspect_candidates_list (list): list of final aspect candidates
        frequency_threshold (int): minimum freq. for qualifying term
    """
    term_polarity = Polarity.UNK
    for extracted_term_list in aspect_candidate_list_raw.values():
        if len(extracted_term_list) >= frequency_threshold:
            term = extracted_term_list[0]
            qualified_term = QualifiedTerm(term.term, term.pos,
                                           len(extracted_term_list),
                                           term_polarity)
            final_aspect_candidates_list.append(qualified_term)

    return final_aspect_candidates_list


def generate_final_opinion_candidates_list(opinion_candidate_list_raw,
                                           final_opinion_candidates_list,
                                           frequency_threshold):
    """
    generate final opinion candidates list from raw opinion candidate list
    Args:
        opinion_candidate_list_raw (dict): key = term, value =
        lists of extracted terms.
        final_opinion_candidates_list (list): list of final opinion candidates
        frequency_threshold (int): minimum freq. for qualifying term
    """
    for candidate_list in opinion_candidate_list_raw.values():
        positive_pol = 0
        negative_pol = 0
        if len(candidate_list) >= frequency_threshold:
            for candidate in candidate_list:
                pol = candidate.term_polarity
                if pol is not None:
                    if pol == Polarity.POS:
                        positive_pol = positive_pol + 1
                    if pol == Polarity.NEG:
                        negative_pol = negative_pol + 1

            term_polarity = Polarity.UNK
            if positive_pol > negative_pol and positive_pol > 0:
                term_polarity = Polarity.POS
            elif negative_pol >= positive_pol and negative_pol > 0:
                term_polarity = Polarity.NEG

            term = candidate_list[0]

            qualified_term = QualifiedTerm(term.term, term.pos,
                                           len(candidate_list), term_polarity)
            final_opinion_candidates_list.append(qualified_term)

    return final_opinion_candidates_list
