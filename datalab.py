import numpy as np
import os


class DataLabTrain(object):

    def __init__(self, path, load_at_a_time=5):
        self.path = path
        self.load_at_a_time = load_at_a_time
        self.li_batch_files = os.listdir(path)
        np.random.shuffle(self.li_batch_files)
        self.cur_index = 0

    def _load_files(self):
        X = []
        Y = []
        for i in range(self.cur_index, self.cur_index + self.load_at_a_time):
            if i == len(self.li_batch_files):
                self.cur_index = -self.load_at_a_time
                break

            file = np.load(self.path+self.li_batch_files[i])
            X.append(file['arr_0'])
            Y.append(file['arr_1'])

        self.cur_index += self.load_at_a_time

        return X, Y

    def generator(self):
        while 1:
            X, Y = self._load_files()
            for i in range(len(X)):
                yield X[i] / 255, Y[i]


class DataLabTest(object):

    def __init__(self, path, load_at_a_time=5):
        self.path = path
        self.load_at_a_time = load_at_a_time
        self.li_batch_files = os.listdir(path)
        self.cur_index = 0

    def _load_files(self):
        X = []
        for i in range(self.cur_index, self.cur_index + self.load_at_a_time):
            if i == len(self.li_batch_files):
                self.cur_index = -self.load_at_a_time
                break

            arr = np.load(self.path+self.li_batch_files[i])
            X.append(arr)

        self.cur_index += self.load_at_a_time

        return X

    def generator(self):
        while 1:
            X = self._load_files()
            for i in range(len(X)):
                yield X[i] / 255
