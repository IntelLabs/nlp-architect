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
from __future__ import print_function

import os
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from functools import reduce

import numpy as np
from ngraph.util.persist import valid_path_append, fetch_file, ensure_dirs_exist
from nlp_architect.utils.generic import license_prompt


def pad_sentences(sentences, sentence_length=None, dtype=np.int32, pad_val=0.):
    lengths = [len(sent) for sent in sentences]

    nsamples = len(sentences)
    if sentence_length is None:
        sentence_length = np.max(lengths)

    X = (np.ones((nsamples, sentence_length)) * pad_val).astype(dtype=np.int32)
    for i, sent in enumerate(sentences):
        trunc = sent[-sentence_length:]

        X[i, :len(trunc)] = trunc
    return X


def pad_stories(stories, sentence_length,  max_story_length, vocab_size, dtype=np.int32,
                pad_val=0., use_time=False):
    nsamples = len(stories)

    X = (np.ones((nsamples, max_story_length, sentence_length)) * pad_val).astype(dtype=np.int32)

    for i, story in enumerate(stories):
        trunc = story[-max_story_length:]
        X[i, :len(trunc)] = trunc

        if use_time:
            for j in range(len(story)):
                X[i, j, -1] = vocab_size - max_story_length + len(story) - j
    return X


def pad_values(values,  max_story_length, dtype=np.int32, pad_val=0.):
    # for values we want an array with our same shape (first value is fine)
    # but filled in with trailing zeros
    nsamples = len(values)
    # len_sample = len(values[0][0])
    len_sample = 1  # temp hack as some questions don't have key-vals

    X = (np.ones((nsamples, max_story_length, len_sample)) * pad_val).astype(dtype=np.int32)

    for i, story in enumerate(values):
        try:
            if len(story) > 0:
                trunc = story[-max_story_length:]
                X[i, :len(trunc)] = trunc
        except:
            pass
    return X


def ex_entity_names(line, dictionary_lookup, regex_list, return_key=False):
    line = line.strip()

    # first try a direct lookup for the knowledge base as it is all that is be necessary for some
    if line in dictionary_lookup:
        e = dictionary_lookup[line]
        if return_key:
            return e, e
        else:
            return e

    for r, e in regex_list:
        try:
            if len(r.findall(line)) > 0:
                line = r.sub(e, line)
                if return_key:
                    return line, e
                else:
                    return line
        except:
            raise

    if return_key:
        return line, None
    else:
        return line


class WIKIMOVIES(object):
    """
    Load WikiMovies dataset and extract text knowledge dictionaries.

    For a particular task, the class will read both train and test files
    and combine the vocabulary.

    Arguments:
        path (str): Directory to store the dataset
    """
    def __init__(self, path='.', subset='wiki-entities', reparse=False,
                 mem_source='kb'):

        self.url = 'http://www.thespermwhale.com/jaseweston/babi'
        self.size = 11745123
        self.filename = 'movieqa.tar.gz'
        self.path = path
        self.reparse = reparse
        data_sub_path = 'movieqa/parsed_data_{}'.format(mem_source)
        data_dict_out_path = valid_path_append(self.path, data_sub_path + '_full_parse.pkl')
        data_dict_train_out_path = valid_path_append(self.path, data_sub_path + '_train.pkl')
        data_dict_test_out_path = valid_path_append(self.path, data_sub_path + '_test.pkl')
        infer_elems_out_path = valid_path_append(self.path, data_sub_path + '_infer_elems.pkl')

        # First try reading from the prevsiously parsed data
        if os.path.exists(data_dict_out_path) and not self.reparse:
            print('Extracting pre-parsed data from ', data_dict_out_path)
            self.data_dict = pickle.load(open(data_dict_out_path, "rb"))
            self.story_length = self.data_dict['info']['story_length']
            self.memory_size = self.data_dict['info']['memory_size']
            self.vocab_size = self.data_dict['info']['vocab_size']

            inference_elems = pickle.load(open(infer_elems_out_path, 'rb'))
            self.full_rev_entity_dict = inference_elems['rev_entity_dict']
            self.full_entity_dict = inference_elems['entity_dict']
            self.knowledge_dict = inference_elems['knowledge_dict']
            self.re_list = inference_elems['regex_list']
            self.word_to_index = inference_elems['word_index']
            self.index_to_word = inference_elems['index_to_word']

        else:
            if not os.path.exists(data_dict_train_out_path) or self.reparse:
                print('Preparing WikiMovies dataset or extracting from %s' % path)
                self.entity_file, self.kb_file, self.train_file, self.test_file = \
                    self.load_data(path, subset=subset)
                print('Creating Entity Dictionary')
                self.full_entity_dict, self.full_rev_entity_dict, self.re_list = \
                    self.create_entity_dict()
                print('Creating knowledge base information')
                if mem_source == 'text':
                    print('Creating knowledge base information from text')
                    self.knowledge_dict = self.parse_text_window()
                else:
                    print('Creating knowledge base information from kb')
                    self.knowledge_dict = self.parse_kb(self.full_rev_entity_dict)

                print('Parsing files')
                self.train_parsed = WIKIMOVIES.parse_wikimovies(self.train_file,
                                                                self.full_rev_entity_dict,
                                                                self.knowledge_dict, self.re_list)
                self.test_parsed = WIKIMOVIES.parse_wikimovies(self.test_file,
                                                               self.full_rev_entity_dict,
                                                               self.knowledge_dict, self.re_list)

                print('Writing to ', data_dict_train_out_path)
                with open(data_dict_train_out_path, 'wb') as f:
                    pickle.dump(self.train_parsed, f)

                with open(data_dict_test_out_path, 'wb') as f:
                    pickle.dump(self.test_parsed, f)

                # Save items needed for inference
                save_elems = {'rev_entity_dict': self.full_rev_entity_dict,
                              'entity_dict': self.full_entity_dict,
                              'knowledge_dict': self.knowledge_dict,
                              'regex_list': self.re_list
                              }
                with open(infer_elems_out_path, 'wb') as f:
                    pickle.dump(save_elems, f)

            else:
                self.data_dict = {}
                self.test_parsed = pickle.load(open(data_dict_test_out_path, 'rb'))
                self.train_parsed = pickle.load(open(data_dict_train_out_path, 'rb'))

            print('Computing Stats')
            self.compute_statistics(self.train_parsed, self.test_parsed)

            print('Vectorizing')
            self.test = self.vectorize_stories(self.test_parsed)
            print('done test')
            self.train = self.vectorize_stories(self.train_parsed)
            print('done train')

            self.story_length = self.story_maxlen
            self.memory_size = self.max_storylen
            self.query_length = self.query_maxlen

            self.data_dict['train'] = {'keys': {'data': self.train[0],
                                                'axes': ('batch', 'memory_axis', 'sentence_axis')},
                                       'values': {'data': self.train[1],
                                                  'axes': ('batch', 'memory_axis', 1)},
                                       'query': {'data': self.train[2],
                                                 'axes': ('batch', 'sentence_axis')},
                                       'answer': {'data': self.train[3],
                                                  'axes': ('batch', 'vocab_axis')}
                                       }

            self.data_dict['test'] = {'keys': {'data': self.test[0],
                                               'axes': ('batch', 'memory_axis', 'sentence_axis')},
                                      'values': {'data': self.test[1],
                                                 'axes': ('batch', 'memory_axis', 1)},
                                      'query': {'data': self.test[2],
                                                'axes': ('batch', 'sentence_axis')},
                                      'answer': {'data': self.test[3],
                                                 'axes': ('batch', 'vocab_axis')}
                                      }

            self.data_dict['info'] = {'story_length': self.story_length,
                                      'memory_size': self.memory_size,
                                      'vocab_size': self.vocab_size
                                      }

            print('Writing to ', data_dict_out_path)
            with open(data_dict_out_path, 'wb') as f:
                pickle.dump(self.data_dict, f)

            # Make sure the index_to_word has an entity UNK for the zero ID
            # This is used during inference
            self.index_to_word[0] = 'UNKNOWN'

            # Also save out the inference elements with the word_to_index
            inference_elems = pickle.load(open(infer_elems_out_path, 'rb'))
            save_elems = {'entity_dict': inference_elems['entity_dict'],
                          'rev_entity_dict': inference_elems['rev_entity_dict'],
                          'knowledge_dict': inference_elems['knowledge_dict'],
                          'regex_list': inference_elems['regex_list'],
                          'word_index': self.word_to_index,
                          'index_to_word': self.index_to_word,
                          }
            with open(infer_elems_out_path, 'wb') as f:
                pickle.dump(save_elems, f)

    def load_data(self, path=".", subset='wiki_entities'):
        """
        Fetch the Facebook WikiMovies dataset and load it to memory.

        Arguments:
            path (str, optional): Local directory in which to cache the raw
                                  dataset.  Defaults to current directory.

        Returns:
            tuple: knowledge base, entity list, training and test files are returned
        """
        self.data_dict = {}
        self.vocab = None
        workdir, filepath = valid_path_append(path, '', self.filename)
        if not os.path.exists(filepath):
            if license_prompt('WikiMovies',
                              self.url,
                              'Creative Commons Attribution 3.0',
                              'https://research.fb.com/downloads/babi/',
                              self.path) is False:
                sys.exit(0)

            fetch_file(self.url, self.filename, filepath, self.size)

        if subset == 'wiki-entities':
            subset_folder = 'wiki_entities'
        else:
            subset_folder = subset

        babi_dir_name = self.filename.split('.')[0]
        file_base = babi_dir_name + '/questions/' + subset_folder + '/' + subset + '_qa_{}.txt'
        train_file = os.path.join(workdir, file_base.format('train'))
        test_file = os.path.join(workdir, file_base.format('test'))

        entity_file_path = babi_dir_name + '/knowledge_source/entities.txt'
        entity_file = os.path.join(workdir, entity_file_path)

        knowledge_file_path = babi_dir_name + '/knowledge_source/' + subset_folder + '/' \
            + subset_folder + '_kb.txt'
        kb_file = os.path.join(workdir, knowledge_file_path)

        return entity_file, kb_file, train_file, test_file

    def create_entity_dict(self):
        # TODO it would be nice to remove the entities films, movies
        # the questions sometimes picks up this entity first
        entity_dict = {}
        reverse_dictionary = {}
        ent_list = []
        with open(self.entity_file) as read:
            for l in read:
                l = l.strip().lower()
                if len(l) > 0:
                    ent_list.append(l)
        ent_list.sort(key=lambda x: -len(x))
        for i in range(len(ent_list)):
            k = ent_list[i]
            v = 'ENTITY_{}'.format(i)
            reverse_dictionary[k] = v
            entity_dict[v] = k
        re_list = [(re.compile('\\b{}\\b'.format(re.escape(e))),
                    '{}'.format(reverse_dictionary[e])) for e in ent_list]
        return entity_dict, reverse_dictionary, re_list

    def parse_text_window(self):
        wiki_window_file = valid_path_append(self.path, 'movieqa/lower_wiki-w=0-d=3-m-4.txt')
        # as it stands the wiki_window_file is created ahead of time...
        # this bit looks for it and completed the pre-processing
        # it can take a couple of hours for this process to complete
        if os.path.exists(wiki_window_file):
            print('Wiki Window Previously created')
        else:
            print('Pre-processing text through wiki windows')
            print('Note that this could take a number of hours')
            subprocess.call(['python', 'wikiwindows.py', self.path])

        knowledge_dict = defaultdict(list)
        with open(wiki_window_file) as read:
            movie_ent = None
            for l in read:
                l = l.strip()
                if len(l) > 0:
                    nid, line = l.split(' ', 1)

                    if nid == '1':
                        movie_ent = line.split(' ')[0]

                    sentence, center = line.split('\t')
                    # Add the movie sentences as well as the center encoding
                    knowledge_dict[movie_ent].append((sentence, center))
                    knowledge_dict[center].append((sentence, movie_ent))

        return knowledge_dict

    def parse_kb(self, reverse_dictionary):
        # Check data repo to see if this exists, if not then run code and save
        workdir, kb_file_path = valid_path_append(self.path, '', 'movieqa/kb_dictionary.pkl')
        if os.path.exists(kb_file_path) and not self.reparse:
            print('Loading files from path')
            print(kb_file_path)
            knowledge_dict = pickle.load(open(kb_file_path, "rb"))
            return knowledge_dict

        action_words = ['directed_by', 'written_by', 'starred_actors', 'release_year',
                        'in_language', 'has_tags', 'has_plot', 'has_imdb_votes', 'has_imdb_rating']
        rev_actions_pre = 'REV_'

        babi_data = open(self.kb_file).read()
        lines = self.data_to_list(babi_data)

        knowledge_dict = defaultdict(list)
        fact = None
        for line in lines:
            if len(line) > 1:
                nid, line = line.lower().split(' ', 1)

                if int(nid) == 1:
                    fact = None

                for a in action_words:
                    # find the action word and split on that, ignoring the has_plot info for now
                    if len(line.split(a)) > 1 and a != 'has_plot':
                        # there can be more than one entity on the left hand side
                        # (particually with starred_actors)
                        entities = line.split(a)
                        # Let's use the info that we know the fact for related stories
                        if not fact:
                            fact = ex_entity_names(entities[0], reverse_dictionary, self.re_list)
                        subject_entities = [ex_entity_names(e_0, reverse_dictionary, self.re_list)
                                            for e_0 in entities[1].split(', ')]

                        # create a hash for the knowledge base where key is the subject
                        # add the fact and reverse fact
                        for subject in subject_entities:
                            knowledge_dict[fact].append((fact + ' ' + a, subject))
                            # Also add reverse here
                            knowledge_dict[subject].append((subject + ' ' +
                                                            rev_actions_pre+a, fact))

        kb_out_path = ensure_dirs_exist(os.path.join(workdir, kb_file_path))

        print('Writing to ', kb_out_path)
        with open(kb_out_path, 'wb') as f:
            pickle.dump(knowledge_dict, f)

        return knowledge_dict

    def reduce_entity_dictionaries(self):
        self.reduced_entity_dict = dict()
        self.reduced_reverse_dict = dict()
        for keeper_key in self.knowledge_dict.keys():
            try:
                self.reduced_entity_dict[keeper_key] = self.full_entity_dict[keeper_key]
                self.reduced_reverse_dict[self.full_entity_dict[keeper_key]] = keeper_key
            except:
                continue
        return self.reduced_entity_dict, self.reduced_reverse_dict

    @staticmethod
    def parse_wikimovies(wikimovies_file, reverse_dictionary, knowledge_dict, regex_list):
        """
        Parse bAbI data into queries, and answers.

        Arguments:
            babi_data (string): String of bAbI data.
            babi_file (string): Filename with bAbI data.

        Returns:
            list of tuples : List of (story, query, answer) words.
        """
        babi_data = open(wikimovies_file).read()
        lines = WIKIMOVIES.data_to_list(babi_data)

        data = []
        for line in lines:
            nid, line = line.lower().split(' ', 1)
            try:
                # Have to actually find the entities before hand as some have question marks...
                q, a = line.split('\t')
                q, key = ex_entity_names(q, reverse_dictionary, regex_list, return_key=True)

                q = WIKIMOVIES.tokenize(q)
                if key in knowledge_dict:
                    related_facts = knowledge_dict[key]
                else:
                    related_facts = []

                # transform each answer into the entities creating a list of all answer entities
                a = WIKIMOVIES.tokenize(' '.join([ex_entity_names(a_0, reverse_dictionary, regex_list)
                                        for a_0 in a.split(',')]))
                data.append((related_facts, q, a))

            except:
                print(line)
                raise

            if len(data) % 1000 == 0:
                # return data
                print(len(data))
        return data

    def compute_statistics(self, train_data, test_data):
        """
        Compute vocab, word index, and max length of stories and queries.
        """
        all_data = train_data + test_data

        all_text = reduce(lambda x, y: x + y, (list(self.flatten_kvs(s)) + q + a
                          for s, q, a in all_data))
        vocab = sorted(set(all_text))
        print(len(vocab))
        # vocab = sorted(reduce(lambda x, y: x | y,
        # (set( list(self.flatten_kvs(s)) + q + a ) for s, q, a in all_data)))
        # Reserve 0 for masking via pad_sequences and self.vocab_size - 1 for <UNK> token
        self.vocab = vocab
        self.vocab_size = len(vocab) + 1
        self.word_to_index = dict((c, i + 1) for i, c in enumerate(vocab))
        self.index_to_word = dict((i + 1, c) for i, c in enumerate(vocab))

        stories = [s for s, _, _ in all_data]
        questions = [q for _, q, _ in all_data]
        flat_stories = [self.flatten(s) for s in stories if s != []]
        max_question_len = max(list(map(len, questions)))
        # longest k/v length
        max_kv_length = max([len(s) for s in self.flatten(flat_stories)])
        # maximum length of stories for an entity (ie number of KV pairs)
        # self.max_storylen = max(list(map(len, stories))) #correct way to calculate
        self.max_storylen = 25  # accounts for >98% of story size
        self.story_maxlen = max(max_kv_length, max_question_len) + 1
        self.vocab_size += self.max_storylen
        print('Dataset Statistics (story lengh, num_stories, vocab size):')
        print(self.max_storylen, self.story_maxlen, self.vocab_size)
        self.query_maxlen = max(list(map(len, (q for _, q, _ in all_data))))

    def words_to_vector(self, words):
        """
        Convert a list of words into vector form.

        Arguments:
            words (list) : List of words.

        Returns:
            list : Vectorized list of words.
        """
        if type(words) != str:
            index_array = []
            for w in words:
                if w in self.word_to_index:
                    index_array.append(self.word_to_index[w])
                else:
                    index_array.append(0)
            return index_array
        else:
            if words in self.word_to_index:
                return [self.word_to_index[words]]
            else:
                return [0]

    def one_hot_vector(self, answer):
        """
        Create one-hot representation of an answer.

        Arguments:
            answer (string) : The word answer.

        Returns:
            list : One-hot representation of answer.
        """
        vector = np.zeros(self.vocab_size)
        try:
            if type(answer) == list:
                for a in answer:
                    vector[self.word_to_index[a]] = 1
            else:
                vector[self.word_to_index[answer]] = 1
        except:
            return vector
        return vector

    def vectorize_stories(self, data):
        """
        Convert (story, query, answer) word data into vectors.

        If sentence length < story_maxlen it is padded with 0's
        If story length < memory size, it is padded with empty memorys (story_length 0's)

        Arguments:
            data (tuple) : Tuple of story, query, answer word data.

        Returns:
            tuple : Tuple of story, query, answer vectors.
        """
        k, v, q, a = [], [], [], []
        for story, query, answer in data:
            # Original paper takes first answer regardless of number of answers
            if len(answer) == 1:
                k.append([self.words_to_vector(sent[0].split()) for sent in story])
                v.append([self.words_to_vector(sent[1]) for sent in story])
                q.append(self.words_to_vector(query))
                a.append(self.words_to_vector(answer)[0])

        k = np.array([pad_sentences(sents, self.story_maxlen) for sents in k])
        k = pad_stories(k, self.story_maxlen, self.max_storylen, len(self.vocab) + 1)

        v = pad_values(v, self.max_storylen)
        q = pad_sentences(q, self.story_maxlen)
        a = np.array(a)
        return (k, v, q, a)

    @staticmethod
    def data_to_list(data):
        """
        Clean a block of data and split into lines.

        Arguments:
            data (string) : String of bAbI data.

        Returns:
            list : List of cleaned lines of bAbI data.
        """
        split_lines = data.split('\n')[:-1]
        return [line.strip() for line in split_lines]

    @staticmethod
    def tokenize(sentence):
        """
        Split a sentence into tokens including punctuation.

        Arguments:
            sentence (string) : String of sentence to tokenize.

        Returns:
            list : List of tokens.
        """
        sentence = sentence.replace('?', '')
        sentence = sentence.replace('.', '')
        return [x.strip() for x in re.split('(\W+)?', sentence) if x.strip()]

    @staticmethod
    def flatten(data):
        """
        Flatten a list of data.

        Arguments:
            data (list) : List of list of words.

        Returns:
            list : A single flattened list of all words.
        """
        if len(data) > 0:
            return reduce(lambda x, y: x + y, data)
        else:
            return data

    @staticmethod
    def flatten_kvs(data):
        """
        Flatten a list of data.

        Arguments:
            data (list) : List of list of words.

        Returns:
            list : A single flattened list of all words.
        """
        return_list = []
        for k, v in data:
            return_list.extend(k.split(' '))
            return_list.append(v)
        return return_list
