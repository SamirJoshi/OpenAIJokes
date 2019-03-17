import unittest
import numpy as np
from src.simple_rnn_model import SimpleRnnModel
from src.dataset import Dataset

class TestSimpleRnnModel(unittest.TestCase):
    def test_preprocess_data_empty(self):
        data = Dataset('test')
        data.jokes = []
        expected_encoded_jokes = []

        model = SimpleRnnModel()
        actual_encoded_jokes = model.preprocess_data(data)
        self.assertLessEqual(actual_encoded_jokes, expected_encoded_jokes)

    def test_preprocess_data(self):
        data = Dataset('test')
        data.jokes = [
            { 'body': 'this is a test' },
            { 'body': 'the' }
        ]
        expected_encoded_jokes = [
            np.array([116, 104, 105, 115, 32, 105, 115, 32, 97, 32, 116, 101, 115, 116]),
            np.array([116, 104, 101]),
        ]

        model = SimpleRnnModel()
        actual_encoded_jokes = model.preprocess_data(data)
        self.assertEqual(len(actual_encoded_jokes), len(expected_encoded_jokes))
        self.assertTrue(np.array_equal(actual_encoded_jokes[0], expected_encoded_jokes[0]))
        self.assertTrue(np.array_equal(actual_encoded_jokes[1], expected_encoded_jokes[1]))

