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
from nlp_architect.models.absa.inference.data_types import Polarity
from nlp_architect.models.absa.train.data_types import OpinionTerm, QualifiedTerm


def set_opinion_term_polarity(terms_list):
    """Set opinion term polarity.

    Args:
        terms_list (list): list of opinion terms
    """
    first = terms_list[0]
    new_term = first.term

    positive_pol = 0
    negative_pol = 0
    curr_polarity = None
    for term in terms_list:
        try:
            curr_polarity = term.term_polarity
        except Exception as e:
            print("extracted_term missing term_polarity: " + str(e))
        if curr_polarity is not None:
            if curr_polarity == Polarity.POS:
                positive_pol = positive_pol + 1
            if curr_polarity == Polarity.NEG:
                negative_pol = negative_pol + 1
    new_term_polarity = Polarity.UNK
    if positive_pol >= negative_pol and positive_pol > 0:
        new_term_polarity = Polarity.POS
    elif negative_pol >= positive_pol and negative_pol > 0:
        new_term_polarity = Polarity.NEG

    return OpinionTerm(new_term, new_term_polarity)


def generate_final_aspect_candidates_list(aspect_candidate_list_raw,
                                          final_aspect_candidates_list,
                                          frequency_threshold):
    """Generate final aspect candidates list from map.

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
            qualified_term = QualifiedTerm(term.term, term.pos, len(extracted_term_list),
                                           term_polarity)
            final_aspect_candidates_list.append(qualified_term)
    return final_aspect_candidates_list


def generate_final_opinion_candidates_list(opinion_candidate_list_raw,
                                           final_opinion_candidates_list,
                                           frequency_threshold):
    """Generate final opinion candidates list from raw opinion candidate list.

    Args:
        opinion_candidate_list_raw (dict): key = term, value =
        lists of extracted terms.
        final_opinion_candidates_list (list): list of final opinion candidates
        frequency_threshold (int): minimum freq. for qualifying term
    """
    for extracted_term_list in opinion_candidate_list_raw.values():
        positive_pol = 0
        negative_pol = 0
        if len(extracted_term_list) >= frequency_threshold:
            for ex_term in extracted_term_list:
                curr_polarity = ex_term.term_polarity
                if curr_polarity is not None:
                    if curr_polarity == Polarity.POS:
                        positive_pol = positive_pol + 1
                    if curr_polarity == Polarity.NEG:
                        negative_pol = negative_pol + 1

            # set polarity according majority vote
            term_polarity = Polarity.UNK
            if positive_pol > negative_pol and positive_pol > 0:
                term_polarity = Polarity.POS
            elif negative_pol >= positive_pol and negative_pol > 0:
                term_polarity = Polarity.NEG

            term = extracted_term_list[0]
            qualified_term = QualifiedTerm(term.term, term.pos, len(extracted_term_list),
                                           term_polarity)
            final_opinion_candidates_list.append(qualified_term)
    return final_opinion_candidates_list
