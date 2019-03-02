import json
import numpy as np
import argparse

class Dataset:
    def __init__(self):
        self.jokes = []
        self.num_jokes = 0
        self.identifier = ''

    def load_from_json(self):
        raise NotImplementedError

    def load_from_npy_file(self):
        raise NotImplementedError

    def write_to_npy_file(self, path=''):
        filename = path + self.identifier + '_dataset.npy'
        np.save(filename, self.jokes)

    def load_dataset(self, filename, isJson='false'):
        """
        If the npy file is found, load from there. Else, load from json.
        If json is not found, throw
        """

        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
