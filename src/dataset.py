
import json
import numpy as np
import argparse
import pprint
import operator


class Dataset:
    def __init__(self, identifier=''):
        self.jokes = []
        self.identifier = identifier

    def __str__(self):
        dataset_info = '************ Dataset Info ************\n'
        dataset_info += 'Identifier: {}\n'.format(self.identifier)
        dataset_info += 'Number of jokes: {}\n'.format(self.num_jokes)
        dataset_info += '********** End Dataset Info **********\n'

        return dataset_info

    @property
    def num_jokes(self):
        return len(self.jokes)

    def load_from_json(self, path, identifier):
        with open(path, 'r', encoding='utf-8') as json_file:
            self.jokes = np.asarray(json.load(json_file))
            self.jokes = np.asarray(list(filter(lambda joke: joke['score'] > 10, self.jokes)))
            print("jokes length:", len(self.jokes))
            # self.jokes = np.concatenate((self.jokes, np.asarray(list(filter(lambda joke: joke['score'] > 500, self.jokes)))))
            # print("jokes length:", len(self.jokes))


    def load_from_npy_file(self, path):
        self.jokes = np.load(path)
        self.starting_words = {}
        for joke in self.jokes:
            starting_word = joke['title'].split(' ', 1)[0]
            if starting_word in self.starting_words:
                self.starting_words[starting_word] += 1
            else:
                self.starting_words[starting_word] = 1

        self.starting_words = list(dict(sorted(self.starting_words.items(), key=operator.itemgetter(1))[-25:]))


    def write_to_npy_file(self, path=''):
        filename = path + self.identifier + '_dataset.npy'
        np.save(filename, self.jokes)


def generate_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('jokes_path',
                        type=str,
                        help='the pathname of the jokes dataset')
    parser.add_argument('--identifier',
                        type=str,
                        default='jokes_dataset',
                        help='an identifier for the dataset')
    parser.add_argument('--create_npy',
                        default=False,
                        action='store_true',
                        help='is the file of jokes a numpy cache')
    parser.add_argument('--display_info',
                        default=False,
                        action='store_true',
                        help='print the stats of the dataset after loading')

    return parser


if __name__ == '__main__':
    parser = generate_parser()
    args = parser.parse_args()
    dataset = Dataset(args.identifier)
    file_ext = args.jokes_path.rpartition('.')[-1]

    if file_ext == 'npy':
        dataset.load_from_npy_file(args.jokes_path)
    else:
        dataset.load_from_json(args.jokes_path, args.identifier)
        if args.create_npy:
            dataset.write_to_npy_file()

    if args.display_info:
        print(dataset)
