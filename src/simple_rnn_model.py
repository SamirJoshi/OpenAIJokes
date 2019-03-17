import numpy as np
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
        raise NotImplementedError

    def generate_model(self):
        raise NotImplementedError

    def generate_joke(self):
        raise NotImplementedError

if __name__ == '__main__':
    dataset = Dataset('wocka dataset')
    dataset.load_from_npy_file('./wocka_dataset.npy')

    model = SimpleRnnModel()
    encoded_jokes = model.preprocess_data(dataset)
