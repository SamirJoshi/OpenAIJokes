import os
import tensorflow as tf
from src.character_model import JokeCharacterModel
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


class CharacterGruModel(JokeCharacterModel):
    def __init__(self, **kwargs):
        super(CharacterGruModel, self).__init__(**kwargs)

    def __str__(self):
        model_info = '******************** Model Info ********************\n'
        model_info += 'Identifier: Character GRU Model\n'
        model_info += 'Token type: Character\n'
        model_info += 'Model type: GRU\n'
        if self.model is not None:
            model_info += 'Built: Yes\n'
            model_summary = []
            self.model.summary(print_fn=lambda x: model_summary.append(x))
            model_info += '\n'.join(model_summary)
            model_info += '\n'
        else:
            model_info += 'Built: No\n'
        model_info += '****************** End Model Info ******************\n'

        return model_info

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
