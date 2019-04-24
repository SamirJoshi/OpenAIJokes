import unittest
import numpy as np
import tensorflow as tf
from src.character_gru_model import CharacterGruModel
from src.dataset import Dataset

class TestCharacterGRUModel(unittest.TestCase):
    def setUp(self):
        self.seq_length    = 10
        self.embedding_dim = 10
        self.batch_size    = 10
        self.buffer_size   = 10000 
        self.num_rnn_units = 64
        self.dropout_rate = 0.0

        self.loss = loss = lambda labels, logits: tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        self.optimizer = tf.train.AdamOptimizer()
        self.gru = CharacterGruModel(
            seq_length=self.seq_length,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            buffer_size=self.buffer_size,
            num_rnn_units=self.num_rnn_units,
            dropout_rate=self.dropout_rate
        )

    def test_encode_decode_empty(self):
        """ Throw error if encode/decode is invoked without preprocessing data """
        with self.assertRaises(ValueError):
            encoded = self.gru.encode_text('hello world')
            decoded = self.gru.decode_text(np.arange(1, 10))


    def test_create_vocab(self):
        """ Ensure that vocab, map data structures, and dataset are correctly populated after invoking create_vocab on simple test case """
        data = Dataset('test')
        data.jokes = [
            { 'body': 'this is a test' },
            { 'body': 'the' }
        ]

        self.gru.create_vocab(data.jokes)

        # gru.vocab is a sorted list of characters
        expected_vocab = ['\x03', ' ', 'a', 'e', 'h', 'i', 's', 't']
        self.assertEqual(self.gru.vocab, expected_vocab)

        # char -> index dictionary and index -> char
        self.assertIsInstance(self.gru.map_char_to_index, dict)
        self.assertIsInstance(self.gru.index_to_char, np.ndarray)
        self.assertEqual(len(self.gru.map_char_to_index.keys()), len(expected_vocab))
        self.assertEqual(len(self.gru.index_to_char), len(expected_vocab))

        # encode_text and decode_text correctly translate text and np.array of characters respectively
        plain_text = 'this is a test'
        encoded = np.array([7, 4, 5, 6, 1, 5, 6, 1, 2, 1, 7, 3, 6, 7])
        encoded_with_termination = np.concatenate((encoded, [0]))
        np.testing.assert_array_equal(self.gru.encode_text(plain_text), encoded)
        np.testing.assert_array_equal(self.gru.encode_text(plain_text, terminate=True), encoded_with_termination)
        self.assertEqual(self.gru.decode_text(encoded), plain_text)


    def test_train_model_without_preprocessing(self):
        """ Throw NotImplementedError when training without generating model""" 
        with self.assertRaises(NotImplementedError):
            
            self.gru.train_model(
                loss_function=self.loss,
                optimizer=self.optimizer,
                num_epochs=5
            )

    def test_generate_model_small(self):
        """ Test expected architecture of small dataset """
        data = Dataset('test')
        data.jokes = [
            {
                "body": "What do you call a cow with no legs?\r\n\r\nGround Beef!",
                "category": "Animal",
                "id": 1,
                "title": "Cow With No Legs"
            },
            {
                "body": "What do you call a cow jumping over a barbed wire fence?\r\n\r\nUtter destruction.",
                "category": "Animal",
                "id": 2,
                "title": "Jumping Cow"
            },
            {
                "body": "What's black and white and red all over?\r\n\r\nA newspaper.",
                "category": "Other / Misc",
                "id": 4,
                "title": "Black, White and Red"
            }
        ]

        self.gru.preprocess_data(data)
        self.gru.generate_model()
        self.gru.train_model(
            loss_function=self.loss,
            optimizer=self.optimizer,
            num_epochs=2
        )

        self.assertEqual(len(self.gru.model.layers), 3)
        self.assertEqual(self.gru.model.layers[1].output_shape, (self.batch_size, None, self.num_rnn_units))
