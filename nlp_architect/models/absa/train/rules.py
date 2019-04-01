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
from nlp_architect.models.absa.train.data_types import CandidateTerm, \
    RelCategory, DepRelationTerm, POS


def rule_1(dep_rel, gov_entry, dep_entry, text):
    """Extract term if rule 1 applies.

    Args:
        dep_rel (DepRelation): Dependency relation.
        gov_entry (DicEntrySentiment): Governor opinion entry.
        dep_entry (DicEntrySentiment): Dependant opinion entry.
        text (str): Sentence text.
    """
    candidate = None
    (anchor_entry, anchor, related) = (gov_entry, dep_rel.gov, dep_rel.dep) \
        if gov_entry else (dep_entry, dep_rel.dep, dep_rel.gov)

    if related.norm_pos == POS.ADJ and dep_rel.rel.startswith('conj'):
        polarity = anchor_entry.polarity
        candidate = CandidateTerm(related, anchor, text, polarity)
    return candidate


def rule_2(dep_rel, relation_list, dep_entry, text):
    """Extract term if rule 2 applies.

    Args:
        dep_rel (DepRelation): Dependency relation.
        relation_list (list of DepRelation): Generic relations between all tokens.
        dep_entry (OpinionTerm): Dependent token.
        text (str): Sentence text.
    """
    candidate = None
    for curr_rt in relation_list:
        if (curr_rt.gov, curr_rt.rel, curr_rt.dep.norm_pos) == \
                (dep_rel.gov, dep_rel.rel, POS.ADJ) and curr_rt.dep != dep_rel.dep:
            candidate = CandidateTerm(curr_rt.dep, dep_rel.dep, text, dep_entry.polarity)
    return candidate


def rule_3(dep_rel, relation_list, text):
    """Extract term if rule 3 applies.

    Args:
        dep_rel (DepRelation): Dependency relation.
        relation_list (list of DepRelation): Generic relations between all tokens.
        text (str): Sentence text.
    """
    candidate = None
    if dep_rel.gov.norm_pos == POS.NN and is_subj_obj_or_mod(dep_rel):
        aspect = expand_aspect(dep_rel.gov, relation_list)
        candidate = CandidateTerm(aspect, dep_rel.dep, text, Polarity.UNK)
    return candidate


def rule_4(dep_rel, relation_list, text):
    """Extract term if rule 4 applies.

    Args:
        dep_rel (DepRelation): Dependency relation.
        relation_list (list of DepRelation): Generic relations between all tokens.
        relation between tokens
        text (str): Sentence text.
    """
    candidate = None
    for curr_rt in relation_list:
        if curr_rt.gov == dep_rel.gov and curr_rt.dep != dep_rel.dep and \
                curr_rt.dep.norm_pos == POS.NN and is_subj_obj_or_mod(curr_rt) and \
                is_subj_obj_or_mod(dep_rel):
            aspect = expand_aspect(curr_rt.dep, relation_list)
            candidate = CandidateTerm(aspect, dep_rel.dep, text, Polarity.UNK)
    return candidate


def rule_5(dep_rel, text):
    """Extract term if rule 5 applies.

    Args:
        dep_rel (DepRelation): Dependency relation.
        text (str): Sentence text.
    """
    candidate = None
    if is_subj_obj_or_mod(dep_rel) and dep_rel.dep.norm_pos == POS.ADJ:
        return CandidateTerm(dep_rel.dep, dep_rel.gov, text, Polarity.UNK)
    return candidate


def rule_6(dep_rel, relation_list, text):
    """Extract term if rule 6 applies.

    Args:
        dep_rel (DepRelation): Dependency relation.
        relation_list (list of DepRelation): Generic relations between all tokens.
        text (str): Sentence text.
    """
    candidate = None
    if dep_rel.rel in ('conj_and', 'conj_but'):
        aspect = expand_aspect(dep_rel.dep, relation_list)
        candidate = CandidateTerm(aspect, dep_rel.gov, text, Polarity.UNK)
    return candidate


def is_subj_obj_or_mod(rt):
    return any(rt.rel in cat.value for cat in (RelCategory.SUBJ, RelCategory.OBJ, RelCategory.MOD))


def expand_aspect(in_aspect_token, relation_list):
    """Expand aspect by Looking for a noun word that it's gov is the aspect. if it has (noun)
    compound relation add it to aspect."""
    aspect = DepRelationTerm(text=in_aspect_token.text, lemma=in_aspect_token.lemma,
                             pos=in_aspect_token.pos, ner=in_aspect_token.ner,
                             idx=in_aspect_token.idx)
    for rel in relation_list:
        if (rel.rel == 'compound') and (rel.gov.idx == aspect.idx):
            diff_positive = aspect.idx - len(rel.dep.text) - 1 - rel.dep.idx
            diff_negative = rel.dep.idx - len(aspect.text) - 1 - aspect.idx
            if diff_positive == 0:
                aspect.text = rel.dep.text + ' ' + aspect.text
                aspect.lemma = rel.dep.text + ' ' + aspect.lemma
                aspect.idx = rel.dep.idx
            if diff_negative == 0:
                aspect.text = aspect.text + ' ' + rel.dep.text
                aspect.lemma = aspect.lemma + ' ' + rel.dep.lemma
    return aspect
