import tensorflow as tf
from src.word_model import JokeWordModel


class WordGruModel(JokeWordModel):
    def __init__(self, **kwargs):
        super(WordGruModel, self).__init__(**kwargs)

    def build(
        self,
        vocab_size,
        embedding_dim,
        num_rnn_units,
        batch_size
    ):
        if tf.test.is_gpu_available():
            rnn = tf.keras.layers.CuDNNGRU
        else:
            rnn = tf.keras.layers.GRU

        layers = [
            tf.keras.layers.Embedding(
                vocab_size,
                embedding_dim,
                batch_input_shape=[batch_size, None]
            ),
            rnn(
                num_rnn_units,
                return_sequences=True,
                recurrent_initializer='glorot_uniform',
                stateful=True
            ),
            tf.keras.layers.Dense(vocab_size)
        ]
        if self.dropout_rate > 0.0 and self.dropout_rate < 1.0:
            layers.insert(2, tf.keras.layers.Dropout(self.dropout_rate))

        return tf.keras.Sequential(layers)

    def generate_model(self):
        self.model = self.build(
            self.vocab_size,
            self.embedding_dim,
            self.num_rnn_units,
            self.batch_size
        )
