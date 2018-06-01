import numpy as np
import os
import urllib.request
from scipy.io import loadmat

SOURCE_URL = "https://github.com/locuslab/TCN/raw/master/TCN/poly_music/mdata/"
JSB_FILENAME = "JSB_Chorales.mat"
Nott_FILENAME = "Nottingham.mat"

class Music():
    def __init__(self, seq_len=100, data_dir="./data", dataset="JSB", batch_size=32):
        self.seq_len = seq_len
        if dataset == "JSB":
            self.filepath = os.path.join(data_dir, JSB_FILENAME)
        if dataset == "Nott":
            self.filepath = os.path.join(data_dir, Nott_FILENAME)
        self._maybe_download(data_dir, dataset)  # list of filepaths

        X_train, X_valid, X_test = self.load_series()

        self.train = self._change_to_seq_len(X_train, self.seq_len + 1)

        self.test = self._change_to_seq_len(X_test + X_valid, self.seq_len + 1)

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

    def load_series(self):
        data = loadmat(self.filepath)
        X_train = list(data['traindata'][0])
        X_valid = list(data['validdata'][0])
        X_test = list(data['testdata'][0])
        return X_train, X_valid, X_test

    def _change_to_seq_len(self, X, seq_len):
        X_padded = np.zeros((len(X), seq_len, X[0].shape[1]))

        for e, x in enumerate(X):
            if x.shape[0] >= seq_len:
                X_padded[e, :, :] = x[-1*seq_len:, :]
            else:
                X_padded[e, -1*x.shape[0]:, :] = x
        return X_padded

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

        if dataset == "JSB":
            filename = JSB_FILENAME
        if dataset == "Nott":
            filename = Nott_FILENAME

        filepath = os.path.join(work_directory, filename)
        req = urllib.request.Request(SOURCE_URL + filename, headers=headers)
        data_handle = urllib.request.urlopen(req)
        with open(filepath, "wb") as fp:
            fp.write(data_handle.read())

        print('Successfully downloaded data to {}'.format(filepath))

        return filepath
