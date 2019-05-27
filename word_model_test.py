import tensorflow as tf
from src.word_gru_model import WordGruModel
from src.dataset import Dataset

dataset = Dataset('wocka dataset')
dataset.load_from_npy_file('./wocka_dataset.npy')

# Model hyperparameters
seq_length    = 30
embedding_dim = 256
batch_size    = 64
buffer_size   = 10000 # Size of buffer used to shuffle dataset
num_rnn_units = 1024
num_epochs    = 5
temperature   = 0.75
dropout_rate = 0.5

gru = WordGruModel(
    seq_length=seq_length,
    embedding_dim=embedding_dim,
    batch_size=batch_size,
    buffer_size=buffer_size,
    num_rnn_units=num_rnn_units,
    dropout_rate=dropout_rate
)

gru.preprocess_data(dataset)
gru.generate_model()

loss = lambda labels, logits: tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)
history = gru.train_model(
    loss_function=loss,
    optimizer=tf.train.AdamOptimizer(),
    num_epochs=num_epochs
)

output = gru.generate_joke(
    start_string="The",
    num_words=100,
    temperature=temperature,
    load_weights=True
)

print(output)
