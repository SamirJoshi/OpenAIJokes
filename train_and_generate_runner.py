import tensorflow as tf
from src.character_gru_model import CharacterLstmModel
from src.dataset import Dataset
import argparse
import random
import pickle

def generate_start_string(dataset):
    return dataset.starting_words[random.randint(0, len(dataset.starting_words)-1)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False, action='store_true', help='train')

    # Model hyperparameters
    parser.add_argument('--seq_length', default=50, type=int, help='Sequence length')
    parser.add_argument('--embedding_dim', default=256, type=int, help='Embedding dimension')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    parser.add_argument('--buffer_size', default=10000, type=int, help='Buffer size')
    parser.add_argument('--num_units', default=256, type=int, help='Number of units per layer')
    parser.add_argument('--num_epochs', default=5, type=int, help='Number of epochs')
    parser.add_argument('--dropout_rate', default=0.0, type=float, help='Dropout rate 0.0 <= dropout_rate < 1.0')

    # Jokes Generation parameters
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature')
    parser.add_argument('--num_chars', default=100, type=int, help='Max number of characters to generate')
    args = parser.parse_args()

    print("train:", args.train)
    print("temperature:", args.temperature)
    print("number of characters:", args.num_chars)

    dataset = Dataset('reddit 10 dataset')
    dataset.load_from_npy_file('./reddit10_dataset.npy')

    gru = CharacterLstmModel(
        seq_length=args.seq_length,
        embedding_dim=args.embedding_dim,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        num_rnn_units=args.num_units,
        dropout_rate=args.dropout_rate
    )

    # gru.preprocess_data(dataset_10_plus, './reddit10_dataset_processed.pickle', True, True)
    gru.preprocess_data(dataset)
    gru.generate_model()
    print(gru)

    if args.train:
        loss = lambda labels, logits: tf.keras.backend.sparse_categorical_crossentropy(labels, logits, from_logits=True)
        history = gru.train_model(
            loss_function=loss,
            optimizer=tf.train.AdamOptimizer(),
            num_epochs=args.num_epochs
        )

    for i in range(0, 10):
        output = gru.generate_joke(
            start_string=generate_start_string(dataset),
            num_characters=args.num_chars,
            temperature=args.temperature,
            load_weights=True
        )

        print('______________________________________________')
        print('\n')
        print(output)
        print('\n')
        print('_____________________________________________')