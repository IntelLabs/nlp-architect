#!/usr/bin/env python
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

from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
import pickle
import itertools
import tarfile
import os
import sys
import numpy as np
from tqdm import tqdm
from nlp_architect.utils.generic import license_prompt
from nlp_architect.utils.io import download_unlicensed_file, valid_path_append


def pad_sentences(sentences, sentence_length=0, pad_val=0.):
    """
    Pad all sentences to have the same length (number of words)
    """
    lengths = [len(sent) for sent in sentences]

    nsamples = len(sentences)
    if sentence_length == 0:
        sentence_length = np.max(lengths)

    X = (np.ones((nsamples, sentence_length)) * pad_val).astype(dtype=np.int32)
    for i, sent in enumerate(sentences):
        trunc = sent[-sentence_length:]
        X[i, :len(trunc)] = trunc
    return X


def pad_stories(stories, sentence_length, max_story_length, pad_val=0.):
    """
    Pad all stories to have the same number of sentences (max_story_length).
    """
    nsamples = len(stories)

    X = (
        np.ones(
            (nsamples,
             max_story_length,
             sentence_length))
        * pad_val).astype(
            dtype=np.int32)

    for i, story in enumerate(stories):
        X[i, :len(story)] = story

    return X


class BABI_Dialog(object):
    """
    This class loads in the Facebook bAbI goal oriented dialog dataset and
    vectorizes them into user utterances, bot utterances, and answers.

    As described in: "Learning End-to-End Goal Oriented Dialog".
    https://arxiv.org/abs/1605.07683.

    For a particular task, the class will read both train and test files
    and combine the vocabulary.

    Args:
        path (str): Directory to store the dataset
        task (str): a particular task to solve (all bAbI tasks are train
            and tested separately)
        oov (bool, optional): Load test set with out of vocabulary entity words
        use_match_type (bool, optional): Flag to use match-type features
        use_time (bool, optional): Add time words to each memory, encoding when the
            memory was formed
        use_speaker_tag (bool, optional): Add speaker words to each memory
            (<BOT> or <USER>) indicating who spoke each memory.
        cache_match_type (bool, optional): Flag to save match-type features after processing
        cache_vectorized (bool, optional): Flag to save all vectorized data after processing

    Attributes:
        data_dict (dict): Dictionary containing final vectorized train, val, and test datasets
        cands (np.array): Vectorized array of potential candidate answers, encoded
        as integers, as returned by BABI_Dialog class. Shape = [num_cands, max_cand_length]
        num_cands (int): Number of potential candidate answers.
        max_cand_len (int): Maximum length of a candidate answer sentence in number of words.
        memory_size (int): Maximum number of sentences to keep in memory at any given time.
        max_utt_len (int): Maximum length of any given sentence / user utterance
        vocab_size (int): Number of unique words in the vocabulary + 2 (0 is reserved for
            a padding symbol, and 1 is reserved for OOV)
        use_match_type (bool, optional): Flag to use match-type features
        kb_ents_to_type (dict, optional): For use with match-type features, dictionary of
            entities found in the dataset mapping to their associated match-type
        kb_ents_to_cand_idxs (dict, optional): For use with match-type features, dictionary
            mapping from each entity in the  knowledge base to the set of indicies in the
            candidate_answers array that contain that entity.
        match_type_idxs (dict, optional): For use with match-type features, dictionary
            mapping from match-type to the associated fixed index of the candidate vector
            which indicated this match type.
    """
    def __init__(self, path='.', task=1, oov=False, use_match_type=False,
                 use_time=True, use_speaker_tag=True, cache_match_type=False,
                 cache_vectorized=False):
        self.url = 'http://www.thespermwhale.com/jaseweston/babi'
        self.size = 3032766
        self.filename = 'dialog-bAbI-tasks.tgz'
        self.path = path
        self.task = task - 1
        self.oov = oov
        self.use_match_type = use_match_type
        self.cache_vectorized = cache_vectorized
        self.match_type_vocab = None
        self.match_type_idxs = None

        self.tasks = [
            'dialog-babi-task1-API-calls-',
            'dialog-babi-task2-API-refine-',
            'dialog-babi-task3-options-',
            'dialog-babi-task4-phone-address-',
            'dialog-babi-task5-full-dialogs-',
            'dialog-babi-task6-dstc2-'
        ]

        print('Preparing bAbI-dialog dataset. Looking in ./%s' % path)

        assert task in range(
            1, 7), "given task is not in the bAbI-dialog dataset"

        print('Task is %s' % (self.tasks[self.task]))

        (self.train_file, self.dev_file, self.test_file, self.cand_file,
         self.kb_file, self.vocab_file, self.vectorized_file) = self.load_data()

        # Parse files into sets of dialogue and user/bot utterance pairs
        self.train_dialog = self.parse_dialog(
            self.train_file, use_time, use_speaker_tag)
        self.dev_dialog = self.parse_dialog(
            self.dev_file, use_time, use_speaker_tag)
        self.test_dialog = self.parse_dialog(
            self.test_file, use_time, use_speaker_tag)

        self.candidate_answers_w = self.load_candidate_answers()

        if self.use_match_type:
            self.kb_ents_to_type = self.load_kb()
            self.kb_ents_to_cand_idxs = self.create_match_maps()
        else:
            self.kb_ents_to_type = None
            self.kb_ents_to_cand_idxs = None

        self.compute_statistics()

        if self.use_match_type:
            self.encode_match_feats()

        if self.cache_vectorized and os.path.exists(self.vectorized_file):
            print('Loading cached vectorized data from: {}'.format(self.vectorized_file))
            with open(self.vectorized_file, 'rb') as f:
                (self.train, self.dev, self.test, self.cands) = pickle.load(f)

        else:
            self.train = self.vectorize_stories(self.train_dialog)
            self.dev = self.vectorize_stories(self.dev_dialog)
            self.test = self.vectorize_stories(self.test_dialog)
            self.cands = self.vectorize_cands(self.candidate_answers_w)

            if self.cache_vectorized:
                print('Caching vectorized data at {}'.format(self.vectorized_file))
                with open(self.vectorized_file, 'wb') as f:
                    pickle.dump((self.train, self.dev, self.test, self.cands), f)

        self.data_dict['train'] = {
            'memory': {
                'data': self.train[0], 'axes': ('batch', 'memory_axis', 'sentence_axis')},
            'memory_mask': {
                'data': self.train[1], 'axes': ('batch', 'memory_axis')},
            'user_utt': {
                'data': self.train[2], 'axes': ('batch', 'sentence_axis')},
            'answer': {
                'data': self.train[3], 'axes': ('batch', 'cand_axis')}}

        self.data_dict['dev'] = {
            'memory': {
                'data': self.dev[0], 'axes': ('batch', 'memory_axis', 'sentence_axis')},
            'memory_mask': {
                'data': self.dev[1], 'axes': ('batch', 'memory_axis')},
            'user_utt': {
                'data': self.dev[2], 'axes': ('batch', 'sentence_axis')},
            'answer': {
                'data': self.dev[3], 'axes': ('batch', 'cand_axis')}}

        self.data_dict['test'] = {
            'memory': {
                'data': self.test[0], 'axes': ('batch', 'memory_axis', 'sentence_axis')},
            'memory_mask': {
                'data': self.test[1], 'axes': ('batch', 'memory_axis')},
            'user_utt': {
                'data': self.test[2], 'axes': ('batch', 'sentence_axis')},
            'answer': {
                'data': self.test[3], 'axes': ('batch', 'cand_axis')}}

        # Add question-specific candidate answers if we are using match-type
        if self.use_match_type:
            self.data_dict['train']['cands_mat'] = {
                'data': self.create_cands_mat('train', cache_match_type),
                'axes': ('batch', 'cand_axis', 'REC')}
            self.data_dict['dev']['cands_mat'] = {
                'data': self.create_cands_mat('dev', cache_match_type),
                'axes': ('batch', 'cand_axis', 'REC')}
            self.data_dict['test']['cands_mat'] = {
                'data': self.create_cands_mat('test', cache_match_type),
                'axes': ('batch', 'cand_axis', 'REC')}

    def load_data(self):
        """
        Fetch and extract the Facebook bAbI-dialog dataset if not already downloaded.

        Returns:
            tuple: training and test filenames are returned
        """
        if self.task < 5:
            self.candidate_answer_filename = 'dialog-babi-candidates.txt'
            self.kb_filename = 'dialog-babi-kb-all.txt'
            self.cands_mat_filename = 'babi-cands-with-matchtype_{}.npy'
            self.vocab_filename = 'dialog-babi-vocab-task{}'.format(self.task + 1) +\
                                  '_matchtype{}.pkl'.format(self.use_match_type)
        else:
            self.candidate_answer_filename = 'dialog-babi-task6-dstc2-candidates.txt'
            self.kb_filename = 'dialog-babi-task6-dstc2-kb.txt'
            self.cands_mat_filename = 'dstc2-cands-with-matchtype_{}.npy'
            self.vocab_filename = 'dstc2-vocab-task{}_matchtype{}.pkl'.format(self.task + 1,
                                                                              self.use_match_type)

        self.vectorized_filename = 'vectorized_task{}.pkl'.format(self.task + 1)

        self.data_dict = {}
        self.vocab = None
        self.workdir, filepath = valid_path_append(
            self.path, '', self.filename)
        if not os.path.exists(filepath):
            if license_prompt('bAbI-dialog',
                              'https://research.fb.com/downloads/babi/',
                              self.path) is False:
                sys.exit(0)

            download_unlicensed_file(self.url, self.filename, filepath, self.size)

        self.babi_dir_name = self.filename.split('.')[0]

        self.candidate_answer_filename = self.babi_dir_name + \
            '/' + self.candidate_answer_filename
        self.kb_filename = self.babi_dir_name + '/' + self.kb_filename
        self.cands_mat_filename = os.path.join(
            self.workdir, self.babi_dir_name + '/' + self.cands_mat_filename)
        self.vocab_filename = self.babi_dir_name + '/' + self.vocab_filename
        self.vectorized_filename = self.babi_dir_name + '/' + self.vectorized_filename

        task_name = self.babi_dir_name + '/' + self.tasks[self.task] + '{}.txt'

        train_file = os.path.join(self.workdir, task_name.format('trn'))
        dev_file = os.path.join(self.workdir, task_name.format('dev'))
        test_file_postfix = 'tst-OOV' if self.oov else 'tst'
        test_file = os.path.join(
            self.workdir,
            task_name.format(test_file_postfix))

        cand_file = os.path.join(self.workdir, self.candidate_answer_filename)
        kb_file = os.path.join(self.workdir, self.kb_filename)
        vocab_file = os.path.join(self.workdir, self.vocab_filename)
        vectorized_file = os.path.join(self.workdir, self.vectorized_filename)

        if (os.path.exists(train_file) is False
                or os.path.exists(dev_file) is False
                or os.path.exists(test_file) is False
                or os.path.exists(cand_file) is False):
            with tarfile.open(filepath, 'r:gz') as f:
                f.extractall(self.workdir)

        return train_file, dev_file, test_file, cand_file, kb_file, vocab_file, vectorized_file

    @staticmethod
    def parse_dialog(fn, use_time=True, use_speaker_tag=True):
        """
        Given a dialog file, parse into user and bot utterances, adding time and speaker tags.

        Args:
            fn (str): Filename to parse
            use_time (bool, optional): Flag to append 'time-words' to the end of each utterance
            use_speaker_tag (bool, optional): Flag to append tags specifiying the speaker to
                each utterance.
        """
        with open(fn, 'r') as f:
            text = f.readlines()

        # Going to be filled with triplets of (memory, last user utterance,
        # desired bot utterance)
        all_dialogues = []
        current_memory = []

        for line in tqdm(text, desc="Parsing"):
            line = line.replace('\n', '')

            # End of dialgue
            if not line:
                current_memory = []
                continue

            number = line.split(' ')[0]

            if '\t' not in line:
                # Line is returned results form API call, store as memory in
                # current dialogue
                current_memory.append(line.split(' ') + ['<USER>'])
            else:
                user_utt, bot_utt = ' '.join(line.split(' ')[1:]).split('\t')

                # Split utterances into words so we can encode as BOW query
                user_utt_w = user_utt.split(' ')
                bot_utt_w = bot_utt.split(' ')

                # Add training example
                # Don't split bot utterance so we can directly compare with
                # candidate answers (and make onehot)
                all_dialogues.append(
                    (current_memory[:], user_utt_w[:], bot_utt))

                # Add time words and speaker tag
                if use_time:
                    user_utt_w += [str(number) + "_TIME"]
                    bot_utt_w += [str(number) + "_TIME"]
                if use_speaker_tag:
                    user_utt_w += ['<USER>']
                    bot_utt_w += ['<BOT>']

                # Add split and modified user and bot utterances to memory
                current_memory += [user_utt_w, bot_utt_w]

        return all_dialogues

    def words_to_vector(self, words):
        """
        Convert a list of words into vector form.

        Args:
            words (list) : List of words.

        Returns:
            list : Vectorized list of words.
        """
        return [self.word_to_index[w] if w in self.vocab else self.word_to_index[
            '<OOV>'] for w in words]

    def one_hot_vector(self, answer):
        """
        Create one-hot representation of an answer.

        Args:
            answer (string) : The word answer.

        Returns:
            list : One-hot representation of answer.
        """
        vector = np.zeros(self.num_cands)
        vector[self.candidate_answers.index(answer)] = 1
        return vector

    def vectorize_stories(self, data):
        """
        Convert (memory, user_utt, answer) word data into vectors.

        If sentence length < max_utt_len it is padded with 0's
        If memory length < memory size, it is padded with empty memorys (max_utt_len 0's)

        Args:
            data (tuple) : Tuple of memories, user_utt, answer word data.

        Returns:
            tuple : Tuple of memories, memory_lengths, user_utt, answer vectors.
        """
        m, ml, m_mask, u, a = [], [], [], [], []
        for mem, utt, answer in tqdm(data, desc="Vectorizing"):
            m.append([self.words_to_vector(sent) for sent in mem])
            ml.append(len(mem))
            mask_zero_len = self.memory_size - ml[-1]
            m_mask.append([1.0 for _ in range(ml[-1])]
                          + [0.0 for _ in range(mask_zero_len)])

            u.append(self.words_to_vector(utt))
            a.append(self.one_hot_vector(answer))

        m = np.array([pad_sentences(sents, self.max_utt_len) for sents in m])
        m = pad_stories(m, self.max_utt_len, self.memory_size)
        m_mask = np.array(m_mask)

        u = pad_sentences(u, self.max_utt_len)
        a = np.array(a)

        return (m, m_mask, u, a)

    def vectorize_cands(self, data):
        """
        Convert candidate answer word data into vectors.

        If sentence length < max_cand_len it is padded with 0's

        Args:
            data (list of lists) : list of candidate answers split into words

        Returns:
            tuple (2d numpy array): padded numpy array of word indexes forr all candidate answers
        """
        c = []
        for cand in data:
            c.append(self.words_to_vector(cand))

        c = pad_sentences(c, self.max_cand_len)
        return c

    def get_vocab(self, dialog):
        """
        Compute vocabulary from the set of dialogs.
        """
        # Extract only the memory words and user utterance words (these will
        # contain all vocab in the end)
        dialog_words = [x[0] + [x[1]] for x in dialog]

        # Concatenate separate dialogues
        all_utts = list(itertools.chain.from_iterable(dialog_words))
        # Concatenate all utterances to get list of words
        all_words = list(itertools.chain.from_iterable(all_utts))

        # Add candidate answer words
        all_words += list(itertools.chain.from_iterable(self.candidate_answers_w))

        if self.use_match_type:
            # Add match-type words
            self.match_type_vocab = list(set(self.kb_ents_to_type.values()))
            all_words += list(self.match_type_vocab) + \
                list(self.kb_ents_to_type.keys())
            # Also compute indicies into each candidate vector for the
            # different match types (fixed position)
            self.match_type_idxs = {
                mt: i + self.max_cand_len_pre_match for i,
                mt in enumerate(
                    self.match_type_vocab)}

        vocab = list(set(all_words))
        return vocab

    def compute_statistics(self):
        """
        Compute vocab, word index, and max length of stories and queries.
        """
        all_dialog = self.train_dialog + self.dev_dialog + self.test_dialog

        self.max_cand_len_pre_match = max(
            list(map(len, self.candidate_answers_w)))

        vocab = self.get_vocab(all_dialog)

        self.vocab = vocab
        # Reserve 0 for masking via pad_sequences, 1 for oov
        self.vocab_size = len(vocab) + 2

        if os.path.exists(self.vocab_file):
            with open(self.vocab_file, 'rb') as f:
                self.word_to_index = pickle.load(f)

            self.index_to_word = dict((self.word_to_index[k], k) for k in self.word_to_index)
        else:
            self.word_to_index = dict((c, i + 2) for i, c in enumerate(vocab))
            self.index_to_word = dict((i + 2, c) for i, c in enumerate(vocab))

            self.word_to_index['<PAD>'] = 0
            self.word_to_index['<OOV>'] = 1
            self.index_to_word[0] = ''  # empty so we dont print a bunch of padding
            self.index_to_word[1] = '<OOV>'

            with open(self.vocab_file, 'wb') as f:
                pickle.dump(self.word_to_index, f)

        memories = [m for m, _, _ in all_dialog]
        self.memory_size = max(list(map(len, memories)))

        dialog_words = [x[0] + [x[1]] for x in all_dialog]
        # Concatenate separate dialogues
        all_utts = list(itertools.chain.from_iterable(dialog_words))
        self.max_utt_len = max(list(map(len, all_utts)))

        self.num_cands = len(self.candidate_answers)

        if self.use_match_type:
            # Add num_match_types slots to each candidate answer so they can
            # be filled if we find a matching word
            self.max_cand_len = self.max_cand_len_pre_match + \
                len(self.match_type_vocab)
        else:
            self.max_cand_len = self.max_cand_len_pre_match

    @staticmethod
    def clean_cands(cand):
        """
        Remove leading line number and final newline from candidate answer
        """
        return ' '.join(cand.split(' ')[1:]).replace('\n', '')

    def load_candidate_answers(self):
        """
        Load candidate answers from file, compute number, and store for final softmax
        """
        with open(self.cand_file, 'r') as f:
            cands_text = f.readlines()
        self.candidate_answers = list(map(self.clean_cands, cands_text))

        # Create BOW representation of candidate answers for final prediction
        # softmax
        candidate_answers_w = list(
            map(lambda x: x.split(' '), self.candidate_answers))
        return candidate_answers_w

    def process_interactive(
            self,
            line_in,
            context,
            response,
            db_results,
            time_feat):
        """
        Parse a given user's input into the same format as training, build the memory
        from the given context and previous response, update the context.
        """
        # Parse user input
        line_in = line_in.replace('\n', '')
        line_w = line_in.split(' ')

        # Enocde user input and current context
        user_utt_w = self.words_to_vector(line_w)

        # Add last bot response to context before we create memory
        if response:
            bot_utt_w = response.split(' ')
            bot_utt_w += [str(time_feat - 1) + "_TIME", '<BOT>']
            context += [bot_utt_w]

        # If line_in is not silence, we are cutting off the api call
        # and making a correction
        if db_results and line_in == '<SILENCE>':
            for result in db_results:
                res_utt_w = [str(time_feat)] + result.split(' ') + ['<USER>']
                context += [res_utt_w]
                time_feat += 1

        # truncate context to max memory size
        context = context[-self.memory_size:]
        memory = [self.words_to_vector(sent) for sent in context]

        # Compute memory mask
        ml = len(memory)
        mask_zero_len = self.memory_size - ml
        m_mask = [1.0 for _ in range(ml)] + [0.0 for _ in range(mask_zero_len)]
        m_mask = np.array(m_mask)

        # Pad memory to max memory fize
        memory = pad_sentences(memory, self.max_utt_len)
        memory_pad = (
            np.zeros(
                (self.memory_size, self.max_utt_len))).astype(
                    dtype=np.int32)
        memory_pad[:len(memory)] = memory

        # Pad user utt to sentence size
        user_utt_pad = (np.zeros((self.max_utt_len,))).astype(dtype=np.int32)
        user_utt_trunc = user_utt_w[-self.max_utt_len:]
        user_utt_pad[:len(user_utt_trunc)] = user_utt_trunc

        # Add time features and store user utterance in context
        user_utt_w_mem = line_w + [str(time_feat) + "_TIME", '<USER>']
        context += [user_utt_w_mem]

        # Compute candidate_matrix if match features, otherwise just return
        # None & it'll use default
        if self.use_match_type:
            # dupliucate candidates for all examples
            cands_mat = np.array(self.cands)

            all_words = list(np.unique(memory.flatten())) + \
                list(np.unique(user_utt_trunc))

            # For each word, if it's an entity, add it's match-type word to the
            # candidate
            for word in all_words:
                if word in self.kb_ents_to_type.keys():
                    cand_idxs = self.kb_ents_to_cand_idxs[word]
                    ent_type_word = self.kb_ents_to_type[word]
                    match_type_word_idx = self.match_type_idxs[ent_type_word]

                    cands_mat[cand_idxs, match_type_word_idx] = ent_type_word
        else:
            cands_mat = np.array(self.cands)

        time_feat += 1

        return user_utt_pad, context, memory_pad, cands_mat, time_feat

    def load_kb(self):
        """
        Load knowledge base from file, parse into entities and types
        """
        with open(self.kb_file, 'r') as f:
            kb_text = f.readlines()

        if self.task < 5:
            # Split each kb entry into entity and entity type match, store as
            # dict
            kb_ents_to_type = {x.strip().split(
                '\t')[-1]: x.strip().split(' ')[2].split('\t')[-2] + "_MATCH" for x in kb_text}
        else:
            kb_ents_to_type = {
                x.strip().split(' ')[3]: x.strip().split(' ')[2] + "_MATCH" for x in kb_text}

        return kb_ents_to_type

    def create_match_maps(self):
        """
        Create dictionary mapping from each entity in the knowledge base to the set of
        indicies in the candidate_answers array that contain that entity. Will be used for
        quickly adding the match type features to the candidate answers during fprop.
        """

        kb_ents = self.kb_ents_to_type.keys()

        kb_ents_to_cand_idxs = {}

        for ent in kb_ents:
            kb_ents_to_cand_idxs[ent] = []

            for c_idx, cand in enumerate(self.candidate_answers_w):
                if ent in cand:
                    kb_ents_to_cand_idxs[ent].append(c_idx)

        return kb_ents_to_cand_idxs

    def encode_match_feats(self):
        """
        Replace entity names and match type names with indexes
        """
        self.kb_ents_to_type = {
            self.word_to_index[k]: self.word_to_index[v] for k,
            v in self.kb_ents_to_type.items()}
        self.kb_ents_to_cand_idxs = {
            self.word_to_index[k]: v for k,
            v in self.kb_ents_to_cand_idxs.items()}
        self.match_type_idxs = {
            self.word_to_index[k]: v for k,
            v in self.match_type_idxs.items()}

    def create_cands_mat(self, data_split, cache_match_type):
        """
        Add match type features to candidate answers for each example in the dataaset.
        Caches once complete.
        """
        cands_mat_filename = self.cands_mat_filename.format(data_split)

        data_dict = self.data_dict[data_split]

        # Returned cached matric if it exists
        if os.path.exists(cands_mat_filename):
            return np.load(cands_mat_filename)

        ndata = data_dict['user_utt']['data'].shape[0]
        # dupliucate candidates for all examples
        cands_mat = np.array([self.cands] * ndata)

        print("Adding match type features to {} set".format(data_split))

        for idx, (mem, utt) in enumerate(
                tqdm(zip(data_dict['memory']['data'], data_dict['user_utt']['data']))):
            # Get list of unique words currently in memory of user utt
            all_words = list(np.unique(mem.flatten())) + list(np.unique(utt))

            # For each word, if it's an entity, add it's match-type word to the
            # candidate
            for word in all_words:
                if word in self.kb_ents_to_type.keys():
                    cand_idxs = self.kb_ents_to_cand_idxs[word]
                    ent_type_word = self.kb_ents_to_type[word]
                    match_type_word_idx = self.match_type_idxs[ent_type_word]

                    cands_mat[idx][
                        cand_idxs, match_type_word_idx] = ent_type_word

        if cache_match_type:
            # Cache computed matrix
            print(
                "Saving candidate matrix for {} to {}".format(
                    data_split,
                    cands_mat_filename))
            np.save(cands_mat_filename, cands_mat)

        return cands_mat
