import unittest
import numpy as np
from src.character_gru_model import CharacterGRUModel
from src.dataset import Dataset

class TestCharacterGRUModel(unittest.TestCase):
    def test_preprocess_data_empty(self):
        data = Dataset('test')
        data.jokes = []
        expected_encoded_jokes = []

        model = CharacterGRUModel()
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

        model = CharacterGRUModel()
        actual_encoded_jokes = model.preprocess_data(data)
        self.assertEqual(len(actual_encoded_jokes), len(expected_encoded_jokes))
        self.assertTrue(np.array_equal(actual_encoded_jokes[0], expected_encoded_jokes[0]))
        self.assertTrue(np.array_equal(actual_encoded_jokes[1], expected_encoded_jokes[1]))

    def test_generate_model_default(self):
        model = CharacterGRUModel()
        model.generate_model()
        self.assertEqual(len(model.model.layers), 3)
        self.assertEqual(model.model.layers[1].output_shape, (None, 1024))

    def test_generate_model_small(self):
        model = CharacterGRUModel(rnn_units=256)
        model.generate_model()
        self.assertEqual(len(model.model.layers), 3)
        self.assertEqual(model.model.layers[1].output_shape, (None, 256))

