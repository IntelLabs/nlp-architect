# -*- coding: utf-8 -*-

"""
Created on 2018/10/6 上午8:52

@author: xujiang@baixing.com

"""
import numpy as np

import tensorflow as tf
from tensorflow import \
    (
        gfile
     )



def load_word2vec(filename, vocab, word_vecs):
    """Loads embeddings in the word2vec binary format which has a header line
    containing the number of vectors and their dimensionality (two integers),
    followed with number-of-vectors lines each of which is formatted as
    '<word-string> <embedding-vector>'.
    Args:
        filename (str): Path to the embedding file.
        vocab (dict): A dictionary that maps token strings to integer index.
            Tokens not in :attr:`vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which is updated as reading from the file.
    Returns:
        The updated :attr:`word_vecs`.
    """
    with gfile.GFile(filename, "rb") as fin:
        header = fin.readline()
        vocab_size, vector_size = [int(s) for s in header.split()]
        if vector_size != word_vecs.shape[1]:
            raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                             (vector_size, word_vecs.shape[1]))
        binary_len = np.dtype('float32').itemsize * vector_size
        for _ in np.arange(vocab_size):
            chars = []
            while True:
                char = fin.read(1)
                if char == b' ':
                    break
                if char != b'\n':
                    chars.append(char)
            word = b''.join(chars)
            word = tf.compat.as_text(word)
            if word in vocab:
                word_vecs[vocab[word]] = np.fromstring(
                    fin.read(binary_len), dtype='float32')
            else:
                fin.read(binary_len)
    return word_vecs

def load_glove(filename, vocab, word_vecs):
    """Loads embeddings in the glove text format in which each line is
    '<word-string> <embedding-vector>'. Dimensions of the embedding vector
    are separated with whitespace characters.
    Args:
        filename (str): Path to the embedding file.
        vocab (dict): A dictionary that maps token strings to integer index.
            Tokens not in :attr:`vocab` are not read.
        word_vecs: A 2D numpy array of shape `[vocab_size, embed_dim]`
            which is updated as reading from the file.
    Returns:
        The updated :attr:`word_vecs`.
    """
    with gfile.GFile(filename) as fin:
        for line in fin:
            vec = line.strip().split()
            if len(vec) == 0:
                continue
            word, vec = vec[0], vec[1:]
            word = tf.compat.as_text(word)
            if word not in vocab:
                continue
            if len(vec) != word_vecs.shape[1]:
                raise ValueError("Inconsistent word vector sizes: %d vs %d" %
                                 (len(vec), word_vecs.shape[1]))
            word_vecs[vocab[word]] = np.array([float(v) for v in vec])
    return word_vecs
