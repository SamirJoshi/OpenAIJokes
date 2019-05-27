import numpy as np
import tensorflow as tf
from os import path
from re import findall
from src.model import JokeBaseModel


class JokeWordModel(JokeBaseModel):
    def __init__(self, **kwargs):
        super(JokeWordModel, self).__init__()
        tf.enable_eager_execution()

        self.num_rnn_units = kwargs['num_rnn_units']
        self.batch_size = kwargs['batch_size']
        self.buffer_size = kwargs['buffer_size']
        self.embedding_dim = kwargs['embedding_dim']
        self.seq_length = kwargs['seq_length']
        self.dropout_rate = kwargs['dropout_rate']
        self.dataset = None
        self.model = None
        self.map_word_to_index = None
        self.index_to_word = None
        self.vocab = None

    @property
    def vocab_size(self):
        return None if self.vocab is None else len(self.vocab)

    def __split_words(self, text):
        # We should find a more descriptive regex that includes conjunctions but excludes singular quotation marks
        # The regex takes a broader approach by accepting conjunctions and lone quotation marks. Quotation marks need to be removed in a second step.
        words = findall(r"[\w']+", text.strip().lower())
        return [word for word in words if word != "''"]

    def create_vocab(self, jokes):
        vocab = set()
        vocab.add(chr(3))

        for joke in jokes:

            words = self.__split_words(joke["body"])
            for word in words:
                vocab.add(word)

        self.vocab = sorted(vocab)
        self.map_word_to_index = {u: i for i, u in enumerate(self.vocab)}
        self.index_to_word = np.array(self.vocab)

    def encode_text(self, text, terminate=False):
        if self.map_word_to_index is None:
            raise ValueError('Model has not been exposed to any data to '
                             'be able to build vocabulary')

        words = self.__split_words(text)

        if terminate:
            words.append(chr(3))

        return np.array([self.map_word_to_index[word] for word in words])

    def decode_text(self, encoded_text):
        return " ".join(
            list(map(lambda x: self.index_to_word[x], encoded_text))
        )

    def preprocess_data(self, data):
        self.create_vocab(data.jokes)
        encoded_joke_text = []
        for joke in data.jokes:
            encoded_joke_text.extend(self.encode_text(joke["body"], terminate=True))

        self.examples_per_epoch = len(encoded_joke_text) // self.seq_length

        # Batch dataset into sequences based on seq_length
        self.sequences = tf.data.Dataset \
            .from_tensor_slices(encoded_joke_text) \
            .batch(self.seq_length + 1, drop_remainder=True)

        # Split dataset into one-off sequences
        self.dataset = self.sequences.map(lambda x: (x[:-1], x[1:]))

    def train_model(
        self,
        loss_function,
        optimizer,
        num_epochs,
        checkpoint_dir="train_checkpoints"
    ):
        if not self.model:
            raise NotImplementedError('Model hasn\'t been built')

        self.model.compile(optimizer=optimizer, loss=loss_function)
        self.dataset = self.dataset \
            .shuffle(self.buffer_size) \
            .batch(self.batch_size, drop_remainder=True)
        checkpoint_prefix = path.join(checkpoint_dir, "ckpt_{epoch}")
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True)

        history = self.model.fit(
            self.dataset.repeat(),
            epochs=num_epochs,
            steps_per_epoch=self.examples_per_epoch // self.batch_size,
            callbacks=[checkpoint_callback])

        return history

    def generate_joke(
        self,
        start_string,
        num_words,
        temperature=1.0,
        load_weights=False,
        checkpoint_dir="train_checkpoints"
    ):
        start_string = start_string.strip()

        # Build model that expects a single batch
        m = self.build(
            self.vocab_size,
            self.embedding_dim,
            self.num_rnn_units,
            1
        )

        if load_weights:
            # Load previously trained weights from the last checkpoint
            m.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        else:
            if not self.model:
                raise ValueError('Model not found. Execute train_model '
                                 'or set load_weights to True to load weights '
                                 'from a previous checkpoint')
            else:
                m.set_weights(self.model.get_weights())

        text_as_index = []
        encoded_start_string = self.encode_text(start_string)
        input_vec = tf.expand_dims(encoded_start_string, axis=0)
        for i in range(num_words):
            predictions = m(input_vec)
            predictions = tf.squeeze(predictions, axis=0)
            predictions = predictions / temperature

            # Draw sample from a distribution of characters
            predicted_index = tf.random.categorical(
                predictions,
                num_samples=1
            )[-1, 0].numpy()

            # Pass predicted character into next input
            input_vec = tf.expand_dims([predicted_index], axis=0)
            text_as_index.append(predicted_index)

            if (self.index_to_word[predicted_index] == chr(3)):
                print('Reached ending character - '
                      'terminating joke generation.')
                break

        return start_string + " " + self.decode_text(text_as_index)

    def generate_model(self):
        raise NotImplementedError()
