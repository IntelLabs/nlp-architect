import gzip
import os

import numpy as np

from neon.data import Dataset, NervanaDataIterator
from neon.data.text_preprocessing import pad_sentences
from utils import get_paddedXY_sequence, get_word_embeddings, sentences_to_ints


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


class CONLL2000(Dataset):
    """
    CONLL 2000 chunking task data set

    Arguments:
        sentence_length (int): number of time steps to embed the data.
        vocab_size (int): max size of vocabulary.
        path (str, optional): Path to data file.
        use_pos (boolean, optional): Yield POS tag features.
        use_chars (boolean, optional): Yield Char RNN features.
        use_w2v (boolean, optional): Use W2V as input features.
        w2v_path (str, optional): W2V model path
    """

    def __init__(self, path='.', sentence_length=50, vocab_size=20000,
                 use_pos=False,
                 use_chars=False,
                 chars_len=20,
                 use_w2v=False,
                 w2v_path=None):
        url = 'https://raw.githubusercontent.com/teropa/nlp/master/resources/corpora/conll2000/'
        self.filemap = {'train': 2842164,
                        'test': 639396}
        self.file_names = ['{}.txt'.format(phase) for phase in self.filemap]
        sizes = [self.filemap[phase] for phase in self.filemap]
        super(CONLL2000, self).__init__(self.file_names,
                                        url,
                                        sizes,
                                        path=path)
        self.sentence_length = sentence_length
        self.vocab_size = vocab_size
        self.use_pos = use_pos
        self.use_chars = use_chars
        self.chars_len = chars_len
        self.use_w2v = use_w2v
        self.w2v_path = w2v_path
        self.vocabs = {}

    def load_gzip(self, filename, size):
        """
        Helper function for downloading test files
        Will download and un-gzip the file into the directory self.path

        Arguments:
            filename (str): name of file to download from self.url
            size (str): size of the file in bytes?

        Returns:
            str: Path to the downloaded dataset.
        """
        _, filepath = self._valid_path_append(self.path, '', filename)

        if not os.path.exists(filepath):
            self.fetch_dataset(self.url, filename, filepath, size)
        if '.gz' in filepath:
            with gzip.open(filepath, 'rb') as fp:
                file_content = fp.readlines()
            filepath = filepath.split('.gz')[0]
            with open(filepath, 'wb') as fp:
                fp.writelines(file_content)
        return filepath

    def load_data(self):
        file_data = {}
        for phase in self.filemap:
            size = self.filemap[phase]
            phase_file = self.load_zip('{}.txt'.format(phase), size)
            file_data[phase] = self.parse_entries(phase_file)
        return file_data['train'], file_data['test']

    @staticmethod
    def parse_entries(filepath):
        texts = []
        block = []
        with open(filepath, 'r', encoding='utf-8') as fp:
            for line in fp:
                if len(line.strip()) == 0:
                    if len(block) > 1:
                        texts.append(list(zip(*block)))
                    block = []
                else:
                    block.append([e.strip() for e in line.strip().split()])
        return texts

    def create_char_features(self, sentences, sentence_length, word_length):
        char_dict = {}
        char_id = 3
        new_sentences = []
        for s in sentences:
            char_sents = []
            for w in s:
                char_vector = []
                for c in w:
                    char_int = char_dict.get(c, None)
                    if char_int is None:
                        char_dict[c] = char_id
                        char_int = char_id
                        char_id += 1
                    char_vector.append(char_int)
                char_vector = [1] + char_vector + [2]
                char_sents.append(char_vector)
            char_sents = pad_sentences(char_sents, sentence_length=word_length)
            if sentence_length - char_sents.shape[0] < 0:
                char_sents = char_sents[:sentence_length]
            else:
                padding = np.zeros(
                    (sentence_length - char_sents.shape[0], word_length))
                char_sents = np.vstack((padding, char_sents))
            new_sentences.append(char_sents)
        char_sentences = np.asarray(new_sentences)
        self.vocabs.update({'char_rnn': char_dict})
        return char_sentences

    def gen_iterators(self):
        train_set, test_set = self.load_data()
        num_train_samples = len(train_set)

        sents = list(zip(*train_set))[0] + list(zip(*test_set))[0]
        X, X_vocab = sentences_to_ints(sents, lowercase=False)
        self.vocabs.update({'token': X_vocab})

        y = list(zip(*train_set))[2] + list(zip(*test_set))[2]
        y, y_vocab = sentences_to_ints(y, lowercase=False)
        self.y_vocab = y_vocab
        X, y = get_paddedXY_sequence(
            X, y, sentence_length=self.sentence_length, shuffle=False)

        self._data_dict = {}
        self.y_size = len(y_vocab) + 1
        train_iters = []
        test_iters = []

        if self.use_w2v:
            w2v_dict, emb_size = get_word_embeddings(self.w2v_path)
            self.emb_size = emb_size
            x_vocab_is = {i: s for s, i in X_vocab.items()}
            X_w2v = []
            for xs in X:
                _xs = []
                for w in xs:
                    if 0 <= w <= 2:
                        _xs.append(np.zeros(emb_size))
                    else:
                        word = x_vocab_is[w - 3]
                        vec = w2v_dict.get(word.lower())
                        if vec is not None:
                            _xs.append(vec)
                        else:
                            _xs.append(np.zeros(emb_size))
                X_w2v.append(_xs)
            X_w2v = np.asarray(X_w2v)
            train_iters.append(TaggedTextSequence(self.sentence_length,
                                                  x=X_w2v[:num_train_samples],
                                                  y=y[:num_train_samples],
                                                  num_classes=self.y_size,
                                                  vec_input=True))
            test_iters.append(TaggedTextSequence(self.sentence_length,
                                                 x=X_w2v[num_train_samples:],
                                                 y=y[num_train_samples:],
                                                 num_classes=self.y_size,
                                                 vec_input=True))
        else:
            train_iters.append(TaggedTextSequence(self.sentence_length,
                                                  x=X[:num_train_samples],
                                                  y=y[:num_train_samples],
                                                  num_classes=self.y_size))
            test_iters.append(TaggedTextSequence(self.sentence_length,
                                                 x=X[num_train_samples:],
                                                 y=y[num_train_samples:],
                                                 num_classes=self.y_size))

        if self.use_pos:
            pos_sents = list(zip(*train_set))[1] + list(zip(*test_set))[1]
            X_pos, X_pos_vocab = sentences_to_ints(pos_sents)
            self.vocabs.update({'pos': X_pos_vocab})
            X_pos, _ = get_paddedXY_sequence(X_pos, y, sentence_length=self.sentence_length,
                                             shuffle=False)
            train_iters.append(TaggedTextSequence(steps=self.sentence_length,
                                                  x=X_pos[:num_train_samples]))
            test_iters.append(TaggedTextSequence(steps=self.sentence_length,
                                                 x=X_pos[num_train_samples:]))

        if self.use_chars:
            char_sentences = self.create_char_features(
                sents, self.sentence_length, self.chars_len)
            char_sentences = char_sentences.reshape(
                -1, self.sentence_length * self.chars_len)
            char_train = char_sentences[:num_train_samples]
            char_test = char_sentences[num_train_samples:]
            train_iters.append(TaggedTextSequence(steps=self.chars_len * self.sentence_length,
                                                  x=char_train))
            test_iters.append(TaggedTextSequence(steps=self.chars_len * self.sentence_length,
                                                 x=char_test))

        if len(train_iters) > 1:
            self._data_dict['train'] = MultiSequenceDataIterator(train_iters)
            self._data_dict['test'] = MultiSequenceDataIterator(test_iters)
        else:
            self._data_dict['train'] = train_iters[0]
            self._data_dict['test'] = test_iters[0]
        return self._data_dict
