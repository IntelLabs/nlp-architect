from __future__ import unicode_literals, print_function, division, \
    absolute_import

import json


class ParsedDocument:
    def __init__(self):
        self.doc_text = None
        self.sentences = []

    def __repr__(self):
        return self.pretty_json()

    def __iter__(self):
        return self.sentences.__iter__()

    def json(self):
        return json.dumps(self.__dict__)

    def pretty_json(self):
        return json.dumps(self.__dict__, indent=4)

    def brat_doc(self):
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

