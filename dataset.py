
import json
import numpy as np
import argparse

class Dataset:
    def __init__(self, identifier=''):
        self.jokes = []
        self.num_jokes = 0
        self.identifier = identifier

    def __str__(self):
        dataset_info = '************ Dataset Info ************\n'
        dataset_info += 'Identifier: {}\n'.format(self.identifier)
        dataset_info += 'Number of jokes: {}\n'.format(self.num_jokes)
        dataset_info += '********** End Dataset Info **********\n'

        return dataset_info

    def load_from_json(self, path):
        with open(path, 'r', encoding='utf-8') as json_file:
            self.jokes = json.load(json_file)

        self.num_jokes = len(self.jokes)

    def load_from_npy_file(self, path):
        raise NotImplementedError

    def write_to_npy_file(self, path=''):
        filename = path + self.identifier + '_dataset.npy'
        np.save(filename, self.jokes)

    def load_dataset(self, filename, isNPyCache=False):
        """
        If the npy file is found, load from there. Else, load from json.
        If json is not found, throw
        """
        if isNPyCache:
            self.load_from_npy_file(filename)
        else:
            self.load_from_json(filename)


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('jokes_path', type=str, help='the pathname of the jokes dataset')
    parser.add_argument('--identifier',
            type=str,
            default='jokes_dataset',
            help='an identifier for the dataset')
    parser.add_argument('--is_npy',
            type=bool,
            default=False,
            help='is the file of jokes a numpy cache')
    parser.add_argument('--create_npy',
            type=bool,
            default=False,
            help='is the file of jokes a numpy cache')
    parser.add_argument('--display_info',
            type=bool,
            default=False,
            help='print te stats of the dataset after loading')

    return parser

if __name__ == '__main__':
    parser = generate_parser()
    args = parser.parse_args()

    dataset = Dataset(args.identifier)
    dataset.load_dataset(args.jokes_path, args.is_npy)

    if (not args.is_npy) and args.create_npy:
        dataset.write_to_npy_file()

    if args.display_info:
        print(dataset)


