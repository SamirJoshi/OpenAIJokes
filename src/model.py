from abc import ABC, abstractmethod


class JokeBaseModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def preprocess_data(self, data):
        pass

    @abstractmethod
    def train_model(self):
        pass

    @abstractmethod
    def generate_model(self):
        pass

    @abstractmethod
    def generate_joke(self):
        pass
