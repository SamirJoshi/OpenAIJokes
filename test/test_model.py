import unittest
from src.model import JokeBaseModel


class TestModel(unittest.TestCase):
    def test_model_base_class_abstract_method_overriding(self):
        """
        Test that abstract class cannot be instantiated and that its
        methods can be overridden by a child class definition
        """
        with self.assertRaises(TypeError):
            _ = JokeBaseModel()
    
        class ChildModel(JokeBaseModel):
            def preprocess_data(self, data):
                return True

            def train_model(self):
                return True

            def generate_model(self):
                return True

            def generate_joke(self):
                return True

        sample_data = [1, 2, 3]
        child = ChildModel()
        self.assertTrue(child.preprocess_data(sample_data))
        self.assertTrue(child.train_model())
        self.assertTrue(child.generate_model())
        self.assertTrue(child.generate_joke())
