import numpy as np
import tensorflow as tf
from os import path
from copy import deepcopy
from src.model import JokeBaseModel


class JokeCharacterModel(JokeBaseModel):
    def __init__(self, **kwargs):
        super(JokeCharacterModel, self).__init__()
        tf.enable_eager_execution()

        self.num_rnn_units = kwargs['num_rnn_units']
        self.batch_size = kwargs['batch_size']
        self.buffer_size = kwargs['buffer_size']
        self.embedding_dim = kwargs['embedding_dim']
        self.seq_length = kwargs['seq_length']
        self.vocab_size = kwargs['vocab_size']
        self.dataset = None
        self.model = None

    @staticmethod
    def encode_text(text):
        return [ord(c) for c in text]

    @staticmethod
    def decode_text(encoded_text):
        return "".join(list(map(chr, encoded_text)))

    @staticmethod
    def build(vocab_size, embedding_dim, num_rnn_units, batch_size):
        raise NotImplementedError()

    def train_model(self, loss_function, optimizer, num_epochs, checkpoint_dir="train_checkpoints"):
        if not self.model:
            raise NotImplementedError('Model hasn\'t been built')

        self.model.compile(optimizer=optimizer, loss=loss_function)
        self.dataset = self.dataset.shuffle(self.buffer_size).batch(self.batch_size, drop_remainder=True)
        
        checkpoint_prefix = path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.model.fit(
            self.dataset.repeat(),
            epochs=num_epochs,
            steps_per_epoch=self.examples_per_epoch // self.batch_size,
            callbacks=[checkpoint_callback])

        return history


    def preprocess_data(self, data):
        encoded_joke_text = []
        for joke in data.jokes:
            encoded_joke_text.extend(self.encode_text(joke['body']))

        self.examples_per_epoch = len(encoded_joke_text) // self.seq_length
        self.sequences = tf.data.Dataset \
            .from_tensor_slices(encoded_joke_text) \
            .batch(self.seq_length + 1, drop_remainder=True)
        split_sequences_and_targets = lambda x: (x[:-1], x[1:])
        self.dataset = self.sequences.map(split_sequences_and_targets)


    def generate_joke(self, start_string, num_characters, temperature=1.0, load_weights=False, checkpoint_dir="train_checkpoints"):
        # Build model that expects a single batch
        m = self.build(self.vocab_size, self.embedding_dim, self.num_rnn_units, 1)

        if load_weights:
            # Load previously trained weights from the last checkpoint
            m.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        else:
            if not self.model:
                raise ValueError("Model not found. Execute train_model or set load_weights to True to load weights from a previous checkpoint")
            else:
                m.set_weights(self.model.get_weights())

        text = []
        encoded_start_string = self.encode_text(start_string)
        input_vec = tf.expand_dims(encoded_start_string, axis=0)
        for i in range(num_characters):
            predictions = m(input_vec)
            predictions = tf.squeeze(predictions, axis=0)
            predictions = predictions / temperature

            # Draws single sample from a categorical distribution of all possible characters
            predicted_index = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

            # Pass predicted character into next input
            input_vec = tf.expand_dims([predicted_index], axis=0)

            print("predicted index: %d, predicted char: %c, predicted value: %f" % (predicted_index, chr(predicted_index), predictions[0, predicted_index]))
            text.append(chr(predicted_index))

        return "".join(text)


class GruCharacterModel(JokeCharacterModel):
    def __init__(self, **kwargs):
        super(GruCharacterModel, self).__init__(**kwargs)

    @staticmethod
    def build(vocab_size, embedding_dim, num_rnn_units, batch_size):
        if tf.test.is_gpu_available():
            rnn = tf.keras.layers.CuDNNGRU
        else:
            rnn = tf.keras.layers.GRU

        return tf.keras.Sequential([
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
        ])

    def generate_model(self):
        self.model = self.build(
            self.vocab_size,
            self.embedding_dim,
            self.num_rnn_units,
            self.batch_size
        )
            

