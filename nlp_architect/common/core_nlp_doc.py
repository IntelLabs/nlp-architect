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


def merge_punct_tok(merged_punct_sentence, last_merged_punct_index, punct_text, is_traverse):
    # merge the text of the punct tok
    if is_traverse:
        merged_punct_sentence[last_merged_punct_index]["text"] = \
            punct_text + merged_punct_sentence[last_merged_punct_index]["text"]
    else:
        merged_punct_sentence[last_merged_punct_index]["text"] = \
            merged_punct_sentence[last_merged_punct_index]["text"] + punct_text


def find_correct_index(orig_gov, merged_punct_sentence):
    for tok_index, tok in enumerate(merged_punct_sentence):
        if tok["start"] == orig_gov["start"] and tok["len"] == orig_gov["len"] and tok["pos"] == \
                orig_gov["pos"] and tok["text"] == orig_gov["text"]:
            return tok_index
    return None


def fix_gov_indexes(merged_punct_sentence, sentence):
    for merged_token in merged_punct_sentence:
        tok_gov = merged_token['gov']
        if tok_gov == -1:  # gov is root
            merged_token['gov'] = -1
        else:
            orig_gov = sentence[tok_gov]
            correct_index = find_correct_index(orig_gov, merged_punct_sentence)
            merged_token['gov'] = correct_index


def merge_punctuation(sentence):
    merged_punct_sentence = []
    tmp_punct_text = None
    punct_text = None
    last_merged_punct_index = -1
    for tok_index, token in enumerate(sentence):
        if token['rel'] == 'punct':
            punct_text = token["text"]
            if tok_index < 1:  # this is the first tok - append to the next token
                tmp_punct_text = punct_text
            else:  # append to the previous token
                merge_punct_tok(merged_punct_sentence, last_merged_punct_index, punct_text,
                                False)
        else:
            merged_punct_sentence.append(token)
            last_merged_punct_index = last_merged_punct_index + 1
            if tmp_punct_text is not None:
                merge_punct_tok(merged_punct_sentence, last_merged_punct_index, punct_text,
                                True)
                tmp_punct_text = None
    return merged_punct_sentence


class CoreNLPDoc:
    """
    Object for core-components (POS, Dependency Relations, etc).

    Args:
        self.doc_text (str): the doc text
        self.sentences (list(list(dict))) : list of sentences, each word in a sentence is
            represented in a dict, structured as follows: {'start': (int), 'len': (int),
            'pos': (str), 'ner': (str), 'lemma': (str), 'gov': (int), 'rel': (str)}
    """

    def __init__(self):
        self.doc_text = ""
        self.sentences = []

    def __repr__(self):
        return self.pretty_json()

    def __iter__(self):
        return self.sentences.__iter__()

    def json(self):
        """
        Return json representations of the object

        Returns:
            :obj:`json`: json representations of the object
        """
        return json.dumps(self.__dict__)

    def pretty_json(self):
        """
        Return pretty json representations of the object

        Returns:
            :obj:`json`: pretty json representations of the object
        """
        return json.dumps(self.__dict__, indent=4)

    def brat_doc(self):
        """
        Return doc adapted to BRAT expected input
        """
        doc = {'text': '', 'entities': [], 'relations': []}
        tok_count = 0
        rel_count = 1
        for sentence in self.sentences:
            sentence_start = sentence[0]['start']
            sentence_end = sentence[-1]['start'] + sentence[-1]['len']
            doc['text'] = doc['text'] + '\n' + self.doc_text[sentence_start:sentence_end]
            token_offset = tok_count

            for token in sentence:
                start = token['start']
                end = start + token['len']
                doc['entities'].append(['T' + str(tok_count), token['pos'], [[start, end]]])

                if token['gov'] != -1 and token['rel'] != 'punct':
                    doc['relations'].append(
                        [rel_count, token['rel'], [['', 'T' + str(token_offset + token['gov'])],
                                                   ['', 'T' + str(tok_count)]]])
                    rel_count += 1
                tok_count += 1
        doc['text'] = doc['text'][1:]
        return doc

    def displacy_doc(self):
        """
        Return doc adapted to displacyENT expected input
        """
        doc = []
        for sentence in self.sentences:
            sentence_doc = {'arcs': [], 'words': []}
            # Merge punctuation:
            merged_punct_sentence = merge_punctuation(sentence)
            fix_gov_indexes(merged_punct_sentence, sentence)
            for tok_index, token in enumerate(merged_punct_sentence):
                sentence_doc['words'].append({'text': token['text'], 'tag': token['pos']})
                dep_tok = tok_index
                gov_tok = token['gov']
                direction = 'left'
                arc_start = dep_tok
                arc_end = gov_tok
                if dep_tok > gov_tok:
                    direction = 'right'
                    arc_start = gov_tok
                    arc_end = dep_tok
                if token['gov'] != -1 and token['rel'] != 'punct':
                    sentence_doc['arcs'].append({'dir': direction, 'label': token['rel'],
                                                 'start': arc_start, 'end': arc_end})
            doc.append(sentence_doc)
        return doc
