import tensorflow as tf
from src.character_model import GruCharacterModel
from src.dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset('wocka dataset')
    dataset.load_from_npy_file('./wocka_dataset.npy')

    # Model hyperparameters
    seq_length    = 15
    embedding_dim = 256
    batch_size    = 64 
    buffer_size   = 10000 # Size of buffer used to shuffle dataset
    vocab_size    = 256   # Size of lexicon from which we draw possible characters
    num_rnn_units = 1024  
    num_epochs    = 5

    gru = GruCharacterModel(
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        buffer_size=buffer_size,
        num_rnn_units=num_rnn_units,
        vocab_size=vocab_size
    )

    gru.preprocess_data(dataset)
    gru.generate_model()

    # Bug: Cannot call train_model and generate_joke in the same process
    # Involves serializing tensors in deepcopy in generate_joke
    # Looking for an alternative to saving weights and generating a new model with the same settings but with a different batch size

    loss = lambda labels, logits: tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    history = gru.train_model(
        loss_function=loss,
        optimizer=tf.train.AdamOptimizer(),
        num_epochs=num_epochs
    )

    output = gru.generate_joke(
        start_string="What ",
        num_characters=100
    )

    # print(output)