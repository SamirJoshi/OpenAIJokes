import tensorflow as tf
from src.character_gru_model import CharacterGruModel
from src.dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset('wocka dataset')
    dataset.load_from_npy_file('./wocka_dataset.npy')

    # Model hyperparameters
    seq_length    = 30
    embedding_dim = 256
    batch_size    = 64 
    buffer_size   = 10000 # Size of buffer used to shuffle dataset
    num_rnn_units = 1024
    num_epochs    = 8
    temperature   = 0.65
    dropout_rate = 0.2

    gru = CharacterGruModel(
        seq_length=seq_length,
        embedding_dim=embedding_dim,
        batch_size=batch_size,
        buffer_size=buffer_size,
        num_rnn_units=num_rnn_units,
        dropout_rate=dropout_rate
    )

    gru.preprocess_data(dataset)
    gru.generate_model()

    print(gru)

    output = gru.generate_joke(
        start_string="The ",
        num_characters=300,
        temperature=temperature,
        load_weights=True
    )

    print(output)