import numpy as np
import tensorflow as tf
from src.model import JokeBaseModel
from src.dataset import Dataset

"""CharacterGRUModel implements a character-level GRU"""
class CharacterGRUModel(JokeBaseModel):
    def __init__(self, rnn_units=1024, batch_size=64):
        super(CharacterGRUModel, self).__init__()
        self.num_rnn_units = rnn_units
        self.batch_size = batch_size
        self.max_joke_length = 1

    def preprocess_data(self, data):
        """ Preprocess the data by encoding the input data at the character
            level

            Args:
                data: Dataset (see Dataset class for details)

            Returns:
                A list of np arrays each of which contains the encoded text
                of the joke body
        """
        self.max_joke_length = 0
        self.vocab_size = 256 # number of ascii characters
        encoded_jokes = []
        for joke in data.jokes:
            joke_text = joke['body']
            if len(joke_text) < 5000:
                encoded_jokes.append(np.array([ord(c) for c in joke['body']]))
                if len(joke_text) > self.max_joke_length:
                    self.max_joke_length = len(joke_text)

        return encoded_jokes

    def create_training_dataset(self, encoded_jokes, seq_length = 20, step_size = 1):
        x_sequences = []
        y_targets = []
        for joke in encoded_jokes[0:1]:
            iteration_length = min(len(joke), self.max_joke_length)
            for i in range(0, iteration_length - seq_length, step_size):
                x_sequences.append(joke[i: i + seq_length])
                y_targets.append(joke[i + seq_length])
        return (x_sequences, y_targets)

    def train_model(self):
        if self.model == None:
            raise TypeError
        raise NotImplementedError

    def get_gru_cell(self):
        """ Choose an GRU layer based on if a GPU is available """
        if tf.test.is_gpu_available():
            return tf.keras.layers.CuDNNGRU
        else:
            return tf.keras.layers.GRU

    def generate_model(self):
        rnn_layer = self.get_gru_cell()
        raise NotImplementedError

    def generate_joke(self):
        raise NotImplementedError

if __name__ == '__main__':
    dataset = Dataset('wocka dataset')
    dataset.load_from_npy_file('./wocka_dataset.npy')

    model = CharacterGRUModel()
    encoded_jokes = model.preprocess_data(dataset)
    (X, y) = model.create_training_dataset(encoded_jokes)
