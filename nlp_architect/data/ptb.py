"""
Data loader for penn tree bank dataset
"""
import os
import numpy as np
import urllib.request


SOURCE_URL = {'PTB': "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz",
              'WikiText-103': "https://s3.amazonaws.com/research.metamind.io/wikitext/" +
                              "wikitext-103-v1.zip"}
FILENAME = {'PTB': "simple-examples", 'WikiText-103': "wikitext-103"}
EXTENSION = {'PTB': "tgz", 'WikiText-103': "zip"}
FILES = {'PTB': lambda x: "data/ptb." + x + ".txt",
         'WikiText-103': lambda x: "wiki." + x + ".tokens"}


class PTBDictionary:
    """
    Class for generating a dictionary of all words in the PTB corpus
    """
    def __init__(self, data_dir="./data", dataset='WikiText-103'):
        """
        Initialize class
        Args:
            data_dir: str, location of data
            dataset: str, name of data corpus
        """
        self.data_dir = data_dir
        self.dataset = dataset
        self.filepath = os.path.join(data_dir, FILENAME[self.dataset])
        self._maybe_download(data_dir)

        self.word2idx = {}
        self.idx2word = []

        self.load_dictionary()
        print("Loaded dictionary of words of size {}".format(len(self.idx2word)))
        self.sos_symbol = self.word2idx['<sos>']
        self.eos_symbol = self.word2idx['<eos>']
        self.save_dictionary()

    def add_word(self, word):
        """
        Method for adding a single word to the dictionary
        Args:
            word: str, word to be added

        Returns:
            None
        """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def load_dictionary(self):
        """
        Populate the corpus with words from train, test and valid splits of data
        Returns:
            None
        """
        for split_type in ["train", "test", "valid"]:
            path = os.path.join(self.data_dir, FILENAME[self.dataset],
                                FILES[self.dataset](split_type))
            # Add words to the dictionary
            with open(path, 'r') as fp:
                tokens = 0
                for line in fp:
                    words = ['<sos>'] + line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.add_word(word)

    def save_dictionary(self):
        """
        Save dictionary to file
        Returns:
            None
        """
        with open(os.path.join(self.data_dir, "dictionary.txt"), "w") as fp:
            for k in self.word2idx:
                fp.write("%s,%d\n" % (k, self.word2idx[k]))

    def _maybe_download(self, work_directory):
        """
        This function downloads the corpus if its not already present
        Args:
            work_directory: str, location to download data to
        Returns:
            None
        """
        if not os.path.exists(self.filepath):
            print("data does not exist, downloading...")
            self._download_data(work_directory)

    def _download_data(self, work_directory):
        """
        This function downloads the corpus
        Args:
            work_directory: str, location to download data to
        Returns:
            None
        """
        work_directory = os.path.abspath(work_directory)
        if not os.path.exists(work_directory):
            os.mkdir(work_directory)

        headers = {'User-Agent': 'Mozilla/5.0'}

        filepath = os.path.join(work_directory, FILENAME[self.dataset] + "." +
                                EXTENSION[self.dataset])
        req = urllib.request.Request(SOURCE_URL[self.dataset], headers=headers)
        data_handle = urllib.request.urlopen(req)
        with open(filepath, "wb") as fp:
            fp.write(data_handle.read())
        print('Successfully downloaded data to {}'.format(filepath))

        if EXTENSION[self.dataset] == "tgz":
            import tarfile
            tar = tarfile.open(filepath, "r:gz")
            tar.extractall(path=work_directory)
            tar.close()
        if EXTENSION[self.dataset] == "zip":
            import zipfile
            zip = zipfile.ZipFile(filepath, 'r')
            zip.extractall(work_directory)
            zip.close()

        print('Successfully unzipped data to {}'.format(os.path.join(work_directory,
                                                                     FILENAME[self.dataset])))

        return


class PTBDataLoader:
    """
    Class that defines data loader
    """
    def __init__(self, word_dict, seq_len=100, data_dir="./data", dataset='WikiText-103',
                 batch_size=32, skip=30, split_type="train", loop=True):
        """
        Initialize class
        Args:
            word_dict: PTBDictionary object
            seq_len: int, sequence length of data
            data_dir: str, location of corpus data
            dataset: str, name of corpus
            batch_size: int, batch size
            skip: int, number of words to skip over while generating batches
            split_type: str, train/test/valid
            loop: boolean, whether or not to loop over data when it runs out
        """
        self.seq_len = seq_len
        self.dataset = dataset

        self.loop = loop
        self.skip = skip

        self.word2idx = word_dict.word2idx
        self.idx2word = word_dict.idx2word

        self.data = self.load_series(os.path.join(data_dir, FILENAME[self.dataset],
                                                  FILES[self.dataset](split_type)))
        self.random_index = np.random.permutation(np.arange(0, self.data.shape[0] - self.seq_len,
                                                            self.skip))
        self.n_train = self.random_index.shape[0]

        self.batch_size = batch_size
        self.sample_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def reset(self):
        """
        Resets the sample count to zero, re-shuffles data
        Returns:
            None
        """
        self.sample_count = 0
        self.random_index = np.random.permutation(np.arange(0, self.data.shape[0] - self.seq_len,
                                                            self.skip))

    def get_batch(self):
        """
        Get one batch of the data
        Returns:
            None
        """
        if self.sample_count + self.batch_size > self.n_train:
            if self.loop:
                self.reset()
            else:
                raise StopIteration("Ran out of data")

        batch_x = []
        batch_y = []
        for _ in range(self.batch_size):
            c_i = int(self.random_index[self.sample_count])
            batch_x.append(self.data[c_i:c_i + self.seq_len])
            batch_y.append(self.data[c_i + 1:c_i + self.seq_len + 1])
            self.sample_count += 1
        batch = (np.array(batch_x), np.array(batch_y))

        return batch

    def load_series(self, path):
        """
        Load all the data into an array
        Args:
            path: str, location of the input data file

        Returns:

        """
        # Tokenize file content
        with open(path, 'r') as fp:
            ids = []
            for line in fp:
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.word2idx[word])

        data = np.array(ids)

        return data

    def decode_line(self, tokens):
        """
        Decode a given line from index to word
        Args:
            tokens: List of indexes

        Returns:
            str, a sentence
        """
        return " ".join([self.idx2word[t] for t in tokens])
