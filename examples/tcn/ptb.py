import numpy as np
import os
import urllib.request


SOURCE_URL = "https://github.com/locuslab/TCN/tree/master/TCN/word_cnn/data/penn"


class PTB:
    def __init__(self, seq_len=100, data_dir="./data", batch_size=32):
        self.seq_len = seq_len

        self.filepath = os.path.join(data_dir, "train.txt")
        self._maybe_download(data_dir)

        self.word2idx = {'<start>': 0, '<eos>': 1, '<pad>': 2}
        self.idx2word = ['<start>', '<eos>', '<pad>']

        self.train = self.load_series(os.path.join(data_dir, "train.txt"))
        self.test = self.load_series(os.path.join(data_dir, "test.txt"))
        self.valid = self.load_series(os.path.join(data_dir, "valid.txt"))

        self.batch_size = batch_size
        self.sample_count = 0
        self.random_train_indices = np.random.permutation(self.train.shape[0])

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def get_batch(self):
        if self.sample_count + self.batch_size > self.train.shape[0]:
            self.sample_count = 0
            self.random_train_indices = np.random.permutation(self.train.shape[0])

        batch = (self.train[self.random_train_indices[self.sample_count: self.sample_count + self.batch_size], 0:self.seq_len, :], self.train[self.random_train_indices[self.sample_count: self.sample_count + self.batch_size], 1:self.seq_len+1, :])
        self.sample_count += self.batch_size
        return batch

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def load_series(self, path):
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = []
            for line in f:
                line_tokens = []
                words = ['<start>'] + line.split() + ['<eos>']
                for ee, word in enumerate(words):
                    if ee >= self.seq_len:
                        break
                    line_tokens.append(self.word2idx[word])
                if ee + 1 < self.seq_len:
                    line_tokens += (self.seq_len - ee - 1) * [self.word2idx['<pad>']]
                ids.append(line_tokens)

        return np.array(ids)

    def _maybe_download(self, work_directory, dataset):
        """
        This function downloads the stock data if its not already present

        Returns:
            Location of saved data

        """
        if (not os.path.exists(self.filepath)):
            print("data does not exist, downloading...")
            self._download_data(work_directory, dataset)

    def _download_data(self, work_directory, dataset):
        work_directory = os.path.abspath(work_directory)
        if not os.path.exists(work_directory):
            os.mkdir(work_directory)

        headers = {'User-Agent': 'Mozilla/5.0'}

        for filename in ["train.txt", "test.txt", "valid.txt"]:
            filepath = os.path.join(work_directory, filename)
            req = urllib.request.Request(SOURCE_URL + filename, headers=headers)
            data_handle = urllib.request.urlopen(req)
            with open(filepath, "wb") as fp:
                fp.write(data_handle.read())
            print('Successfully downloaded data to {}'.format(filepath))

        return
