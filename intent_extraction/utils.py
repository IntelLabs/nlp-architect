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
import os
import posixpath
import random
import string
import subprocess
import tempfile
import zipfile

import numpy as np
import requests
from keras.callbacks import Callback
from tqdm import tqdm


def one_hot(mat, num_classes):
    """
    Convert a 1D matrix of ints into one-hot encoded vectors.

    Arguments:
        mat(numpy.ndarray): A 1D matrix of labels (int)
        num_classes(int): Number of all possible classes

    Returns:
        numpy.ndarray: A 2D matrix
    """
    assert len(mat.shape) < 2 or isinstance(mat.shape, int)
    vec = np.zeros((mat.shape[0], num_classes))
    for i, v in enumerate(mat):
        vec[i][v] = 1.0
    return vec


def one_hot_sentence(mat, num_classes):
    """
    Convert a 2D matrix of ints into one-hot encoded 3D matrix

    Arguments:
        mat(numpy.ndarray): A 2D matrix of labels (int)
        num_classes(int): Number of all possible classes

    Returns:
        numpy.ndarray: A 3D matrix
    """
    new_mat = []
    for i in range(mat.shape[0]):
        new_mat.append(one_hot(mat[i], num_classes))
    return np.asarray(new_mat)


def add_offset(mat, offset=1):
    """
    Add +1 to all values in matrix mat

    Arguments:
        mat(numpy.ndarray): A 2D matrix with int values
        offset(int): offset to add

    Returns:
        numpy.ndarray: input matrix
    """
    for i, vec in enumerate(mat):
        offset_arr = np.array(vec.shape)
        offset_arr.fill(offset)
        mat[i] = vec + offset_arr
    return mat


def load_word_embeddings(file_path, emb_size):
    """
    Loads a word embedding model into a word->vector dict

    Arguments:
        file_path(str): path to model
        emb_size(int): embedding vectors size

    Returns:
        list: list of numpy.ndarray
    """
    with open(file_path) as fp:
        word_vectors = {}
        for line in fp:
            line_fields = line.split()
            if len(line_fields) == emb_size:
                # probably space vec embedding, skip
                continue
            if len(line_fields) > 2:
                assert len(line_fields[1:]) == emb_size, \
                    'Word embedding size (%d) not equal to given size (%d)' % (
                        len(line_fields[1:]), emb_size)
                word_vectors[line_fields[0]] = np.asarray(
                        line_fields[1:], dtype='float32')
    return word_vectors


def fill_embedding_mat(src_mat, src_lex, emb_lex, emb_size):
    """
    Creates a new matrix from given matrix of int words using the embedding
    model provided.
    """
    emb_mat = np.zeros((src_mat.shape[0], src_mat.shape[1], emb_size))
    for i, sen in enumerate(src_mat):
        for j, w in enumerate(sen):
            if w > 0:
                w_emb = emb_lex.get(str(src_lex.get(w)).lower())
                if w_emb is not None:
                    emb_mat[i][j] = w_emb
    return emb_mat


def run_conlleval(filename):
    """
    Run CoNLL benchmarking script to get global and per
    label accuracy, precision, recall and F1.
    Input is a filename in IOB format:
    <TOKEN> - <TEST LABEL> <PREDICTED LABEL>
    Argument:
        filename(str): tagged file path
    Returns:
        (int, int, int), tuple: Overall precision/recall/f1 and per
        label P/R/F1 in tuple (dictionary format)
    """
    if not os.path.exists('conlleval.pl'):
        download_file('https://www.clips.uantwerpen.be/conll2000/chunking/',
                      'conlleval.txt', 'conlleval.pl')
    proc = subprocess.Popen(['perl', 'conlleval.pl'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    stdout, _ = proc.communicate(open(filename, 'rb').read())
    lines = stdout.decode('utf-8').split('\n')[:-1]
    _, _, _, gp, _, gr, _, gf1 = lines[1].split()
    gp = float(gp[:-2])
    gr = float(gr[:-2])
    gf1 = float(gf1)
    labels = {}
    for l in lines[2:]:
        label, _, p, _, r, _, f1, _ = l.split()
        label = label[:-1]
        p = float(p[:-2])
        r = float(r[:-2])
        f1 = float(f1)
        labels[label] = (p, r, f1)
    return (gp, gr, gf1), labels


def get_conll_scores(predictions, y, y_lex):
    if type(predictions) == list:
        predictions = predictions[-1]
    test_p = predictions.argmax(2)
    test_y = y.argmax(2)

    prediction_data = []
    for n in range(test_y.shape[0]):
        test_yval = [y_lex[i] for i in test_y[n] if i > 0]
        prediction_y = ['O'] * len(test_yval)
        for i, j in enumerate(test_p[n][-len(test_yval):]):
            if j > 0:
                prediction_y[i] = y_lex[j]
        prediction_data.append((test_yval, test_yval, prediction_y))

    temp_fname = tempfile.gettempdir() + \
        os.sep + \
        ''.join([random.choice(string.ascii_letters) for _ in range(10)]) + '__conll_eval'
    with open(temp_fname, 'w') as fp:
        for sample in prediction_data:
            for t, l, p in zip(*sample):
                fp.write('{} - {} {}\n'.format(t, l, p))
            fp.write('\n')

    # run CoNLL benchmark
    scores = run_conlleval(temp_fname)
    os.remove(temp_fname)
    return scores


def download_file(url, sourcefile, destfile, totalsz=None):
    """
    Download the file specified by the given URL.
    """
    req = requests.get(posixpath.join(url, sourcefile),
                       stream=True)

    chunksz = 1024 ** 2
    if totalsz is None:
        if "Content-length" in req.headers:
            totalsz = int(req.headers["Content-length"])
            nchunks = totalsz // chunksz
        else:
            print("Unable to determine total file size.")
            nchunks = None
    else:
        nchunks = totalsz // chunksz

    print("Downloading file to: {}".format(destfile))
    with open(destfile, 'wb') as f:
        for data in tqdm(req.iter_content(chunksz), total=nchunks, unit="MB"):
            f.write(data)
    print("Download Complete")


def unzip_file(filepath):
    """
    Unzip a file to the same location of filepath
    """
    z = zipfile.ZipFile(filepath, 'r')
    z.extractall('.')
    z.close()


class ConllCallback(Callback):
    """
    Conlleval evaluator with keras callback support.
    Runs the conlleval script for given x and y inputs.
    Prints Conlleval F1 score.

    Arguments:
        x: features matrix
        y: labels matrix
        y_vocab(dict): int-to-str labels lexicon
    """

    def __init__(self, x, y, y_vocab, batch_size=1):
        super(ConllCallback, self).__init__()
        self.x = x
        self.y = y
        self.y_vocab = {v: k for k, v in y_vocab.items()}
        self.bsz = batch_size

    def on_epoch_end(self, epoch, logs=None):
        predictions = self.model.predict(self.x, batch_size=self.bsz)
        f1 = get_conll_scores(predictions, self.y, self.y_vocab)[0][-1]
        print()
        print('Conll eval F1: {}'.format(f1))


class Vocabulary:
    """
    Object that maps words to ints (storing a vocabulary)
    """

    def __init__(self):
        self._vocab = {}
        self._rev_vocab = {}
        self.next = 0

    def add(self, word):
        if word not in self._vocab.keys():
            self._vocab[word] = self.next
            self._rev_vocab[self.next] = word
            self.next += 1
        return self._vocab.get(word)

    def word_id(self, word):
        return self._vocab.get(word, None)

    def __len__(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    def add_vocab_offset(self, offset):
        new_vocab = {}
        for k, v in self.vocab.items():
            new_vocab[k] = v + offset
        self.next += offset
        self._vocab = new_vocab
        self._rev_vocab = {v: k for k, v in new_vocab.items()}

    def reverse_vocab(self):
        return self._rev_vocab
