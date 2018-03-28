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
import numpy as np
from neon.data.text_preprocessing import pad_sentences


def label_precision_recall_f1(truth, preds, labels):
    """
    Get precision/recall/f1 scores of given truth and predicted labels.
    Calculate metrics only of given list of labels.
    """
    fp = 0.
    fn = 0.
    tp = 0.
    for t, p in zip(truth, preds):
        for t_i, p_i in zip(t, p):
            if t_i in labels and p_i in labels and t_i == p_i:
                tp += 1.
            elif t_i in labels and p_i not in labels or \
                    t_i in labels and p_i in labels and t_i != p_i:
                fn += 1.
            elif t_i not in labels and p_i in labels:
                fp += 1.
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def extract_nps(text, annotation):
    """
    Extract Noun Phrases from given text tokens and phrase annotations
    """
    np_starts = [i for i in range(len(annotation)) if annotation[i] == 'B-NP']
    np_indexes = []
    for s in np_starts:
        i = 1
        while s+i < len(annotation) and annotation[s + i] == 'I-NP':
            i += 1
        np_indexes.append((s, s + i))
    return [' '.join(text[s:e]) for s, e in np_indexes]


def get_paddedXY_sequence(X, y, vocab_size=20000, sentence_length=100, oov=2,
                          start=1, index_from=3, seed=113, shuffle=True):
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

    if start is not None:
        X = [[start] + [w + index_from for w in x] for x in X]
    else:
        X = [[w + index_from for w in x] for x in X]

    if not vocab_size:
        vocab_size = max([max(x) for x in X])

    # word ids - pad (0), start (1), oov (2)
    if oov is not None:
        X = [[oov if w >= vocab_size else w for w in x] for x in X]
    else:
        X = [[w for w in x if w < vocab_size] for x in X]

    X = pad_sentences(X, sentence_length=sentence_length)

    y = [[w + 1.0 for w in i] for i in y]
    y = pad_sentences(y, sentence_length=sentence_length)

    return X, y


def get_paddedXY(X, y, vocab_size=20000, sentence_length=100, oov=2,
                 start=1, index_from=3, seed=113, shuffle=True):

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)

    if start is not None:
        X = [[start] + [w + index_from for w in x] for x in X]
    else:
        X = [[w + index_from for w in x] for x in X]

    if not vocab_size:
        vocab_size = max([max(x) for x in X])

    # word ids - pad (0), start (1), oov (2)
    if oov is not None:
        X = [[oov if w >= vocab_size else w for w in x] for x in X]
    else:
        X = [[w for w in x if w < vocab_size] for x in X]

    X = pad_sentences(X, sentence_length=sentence_length)
    y = np.array(y, dtype=np.int32).reshape((len(y), 1))

    return X, y


def get_word_embeddings(fname, skip_first=False):
    """
    Load word2vec model from supplied path.
    returns the model dictionary and embedding vector size.
    """
    word = ''
    with open(fname, encoding='utf-8') as fp:
        model = {}
        for line in fp:
            if skip_first:
                skip_first = False
                continue
            splitLine = line.split()
            word = splitLine[0]
            model[word] = np.asarray(splitLine[1:], dtype='float32')
        print("Done.", len(model), " words loaded!")
    return model, len(model.get(word))


def sentences_to_ints(texts, lowercase=True):
    """
    convert text sentences into int id sequences. Word ids are sorted
    by frequency of appearance.
    return int sequences and vocabulary.
    """
    w_dict = {}
    for sen in texts:
        for w in sen:
            if lowercase:
                w = w.lower()
            w_dict.update({w: w_dict.get(w, 0) + 1})
    int_to_word = [(i, word[0]) for i, word in
                   enumerate(sorted(w_dict.items(), key=lambda x: x[1], reverse=True))]
    vocab = {w: i for i, w in int_to_word}
    return [[vocab[w.lower()] if lowercase else vocab[w]
             for w in sen] for sen in texts], vocab
