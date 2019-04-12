from src.character_gru_model import CharacterGRUModel
from src.dataset import Dataset

if __name__ == "__main__":
    dataset = Dataset('wocka dataset')
    dataset.load_from_npy_file('./wocka_dataset.npy')

    model = CharacterGRUModel()
    encoded_jokes = model.preprocess_data(dataset)
    (X, y) = model.create_training_dataset(encoded_jokes[0:1], 25)
