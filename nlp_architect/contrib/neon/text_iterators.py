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

# pylint: skip-file
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import

import numpy as np
from neon.data import NervanaDataIterator


class TaggedTextSequence(NervanaDataIterator):
    """
    This class defines methods for loading and iterating over text datasets
    for tagging tasks (POS, NER, Chunking, ...).
    """

    def __init__(self, steps, x, y=None, num_classes=None, vec_input=False):
        """
        Construct a text tagging dataset object.

        Arguments:
            steps (int) : Length of a sequence (sentence).
            x (numpy.ndarray, shape: [# examples, steps]): Input sentences of dataset
                encoded in ints or as flat vectors (if vec_input=True).
            y (numpy.ndarray, shape: [# examples, steps]): Output sentence tags.
            num_classes (int, optional): Number of features in y (# of possible tags of y).
            vec_input (bool, optional): Toggle vectorized input instead of scalars
                for input steps.
        """
        super(TaggedTextSequence, self).__init__(name=None)
        self.batch_index = 0
        self.X_features = self.nclass = x.shape[1]
        self.vec_input = vec_input
        self.nsamples = x.shape[0]
        if self.vec_input:
            self.shape = (x.shape[2], steps)
            self.x_dev = self.be.iobuf((x.shape[2], steps))
        else:
            self.shape = (steps, 1)
            self.x_dev = self.be.iobuf(steps)

        extra_examples = self.nsamples % self.be.bsz

        if y is not None:
            if num_classes is not None:
                self.y_nfeatures = num_classes
            else:
                self.y_nfeatures = y.max() + 1

            self.y_dev = self.be.iobuf((self.y_nfeatures, steps))
            self.y_labels = self.be.iobuf(steps, dtype=np.int32)
            self.dev_lblflat = self.y_labels.reshape((1, -1))

        if extra_examples:
            x = x[:-extra_examples]
            if y is not None:
                y = y[:-extra_examples]

        self.nsamples -= extra_examples
        self.nbatches = self.nsamples // self.be.bsz
        self.ndata = self.nbatches * self.be.bsz * steps

        self.y = None
        if y is not None:
            self.y_series = y
            self.y = y.reshape(self.be.bsz, self.nbatches, steps)

        if self.vec_input:
            self.X = x.reshape(self.be.bsz, self.nbatches, steps, x.shape[2])
        else:
            self.X = x.reshape(self.be.bsz, self.nbatches, steps)

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        """
        self.batch_index = 0

    def __iter__(self):
        """
        Generator that can be used to iterate over this dataset.

        Yields:
            tuple : the next minibatch of x data and y is preset.
        """
        self.batch_index = 0
        while self.batch_index < self.nbatches:
            x_batch = self.X[:, self.batch_index].T.astype(
                np.float32, order='C')
            if self.vec_input:
                x_batch = x_batch.reshape(self.X.shape[-1], -1)
            self.x_dev.set(x_batch)

            if self.y is not None:
                y_batch = self.y[:, self.batch_index].T.astype(
                    np.float32, order='C')
                self.y_labels.set(y_batch)
                self.y_dev[:] = self.be.onehot(self.dev_lblflat, axis=0)

            self.batch_index += 1
            if self.y is not None:
                yield self.x_dev, self.y_dev
            else:
                yield self.x_dev, None


class MultiSequenceDataIterator(NervanaDataIterator):
    """
    Multiple source data iterator

    this iterator combines several data iterators with given y
    and iterates concurrently on each iterator. If y is not given
    it is taken from the first iterator.
    output shape: tuple of elements from each iterator, y element.
    """

    def __init__(self, data_iterators, y=None, ignore_y=False):
        super(MultiSequenceDataIterator, self).__init__()
        assert len(data_iterators) > 1, "data input is not of size > 1"
        self.input_sources = data_iterators
        nbs = {d.nbatches for d in data_iterators}
        assert len(nbs) == 1, "num of batches in data iterators not equal"
        self.nbatches = nbs.pop()
        if y:
            self.y = y
        elif hasattr(data_iterators[0], 'y') and data_iterators[0].y is not None:
            self.y = data_iterators[0].y
        else:
            self.y = None
        self.ignore_y = ignore_y
        self.ndata = data_iterators[0].ndata
        self.shape = [(i.shape[0], i.shape[1] if len(i.shape) > 1 else 1)
                      for i in data_iterators]
        self.iterators = None
        self.y_iter = None

    def reset(self):
        """
        For resetting the starting index of this dataset back to zero.
        """
        for i in self.input_sources:
            i.reset()
        self.iterators = [i.__iter__() for i in self.input_sources]
        if self.y is not None and hasattr(self.y, 'reset') and not self.ignore_y:
            self.y.reset()
            self.y_iter = self.y.__iter__()

    def __iter__(self):
        self.reset()
        for _ in range(self.nbatches):
            x_iters = [next(i) for i in self.iterators]
            if hasattr(self.y, 'y_iter'):
                yield ([x[0] for x in x_iters]), (next(self.y_iter))
            elif self.y is not None:
                yield ([x[0] for x in x_iters]), (x_iters[0][1])
            elif self.ignore_y:
                yield ([x[0] for x in x_iters]), ()
