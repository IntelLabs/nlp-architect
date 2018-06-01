import numpy as np


class Adding:
    def __init__(self, T=200, n_train=50000, n_test=1000, batch_size=32):
        self.T = T
        self.n_train = n_train
        self.n_test = n_test

        X_train, y_train = self.load_data(n_train)
        X_val, y_val = self.load_data(n_test)

        self.train = (X_train, y_train)

        self.test = (X_val, y_val)

        self.sample_count = 0
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_batch()

    def get_batch(self):
        if self.sample_count + self.batch_size > self.n_train:
            self.sample_count = 0

        batch = (self.train[0][self.sample_count: self.sample_count + self.batch_size], self.train[1][self.sample_count: self.sample_count + self.batch_size])
        self.sample_count += self.batch_size
        return batch

    def load_data(self, N):
        """
        Args:
            N: # of data in the set
        """
        X_num = np.random.rand(N, self.T, 1)
        X_mask = np.zeros((N, self.T, 1))
        y = np.zeros((N, 1))
        for i in range(N):
            positions = np.random.choice(self.T, size=2, replace=False)
            X_mask[i, positions[0], 0] = 1
            X_mask[i, positions[1], 0] = 1
            y[i, 0] = X_num[i, positions[0], 0] + X_num[i, positions[1], 0]
        X = np.concatenate((X_num, X_mask), axis=2)
        return X, y