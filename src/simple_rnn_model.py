import numpy as np
import tensorflow as tf
import functools
from src.model import JokeBaseModel
from src.dataset import Dataset

"""SimpleRnnModel implements a character-level RNN"""
class SimpleRnnModel(JokeBaseModel):
    def preprocess_data(self, data):
        """ Preprocess the data by encoding the input data at the character
            level

            Args:
                data: Dataset (see Dataset class for details)

            Returns:
                A list of np arrays each of which contains the encoded text
                of the joke body
        """
        encoded_jokes = []
        for joke in data.jokes:
            encoded_jokes.append(np.array([ord(c) for c in joke['body']]))

        return encoded_jokes

    def train_model(self):
        if self.model == None:
            raise TypeError
        raise NotImplementedError

    def get_rnn_cell(self):
        if tf.test.is_gpu_available():
            return tf.keras.layers.CudnnRNNRelu
        else:
            return functools.partial(tf.keras.layers.RNN, recurrent_activation='relu')

    def generate_model(self):
        vocab_size = 0
        embedding_dim = 256
        rnn_units = 1024
        batch_size = 64
        batch_input_shape = [batch_size, None]
        rnn_layer = self.get_rnn_cell()

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim,
            batch_input_shape=batch_input_shape))
        self.model.add(rnn_layer(rnn_units, return_sequences=True,
            recurrent_initializer='glorot_uniform', stateful=True))
        self.model.add(tf.keras.layers.Dense(vocab_size))
        self.model.compile(optimizer = tf.train.AdamOptimizer(),
            loss='categorical_crossentropy')

    def generate_joke(self):
        raise NotImplementedError

if __name__ == '__main__':
    dataset = Dataset('wocka dataset')
    dataset.load_from_npy_file('./wocka_dataset.npy')

    model = SimpleRnnModel()
    encoded_jokes = model.preprocess_data(dataset)
