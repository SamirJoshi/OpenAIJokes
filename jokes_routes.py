from flask import Flask
from flask_cors import CORS
from train_and_generate_runner import joke_generator, define_model, generate_start_string
from src.dataset import Dataset

app = Flask(__name__)
CORS(app)

model = define_model()
dataset = Dataset('reddit 10 dataset')
dataset.load_from_npy_file('./reddit10_dataset.npy')

@app.route("/")
def generate_joke():
    start_string = generate_start_string(dataset)
    joke = joke_generator(0.5, 200, model, start_string)
    return joke