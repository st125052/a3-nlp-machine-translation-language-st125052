from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import pickle
from pathlib import Path

from helper.translate import *
from classes.transformer.all_classes import *

# Load the model and the scaler
base_dir = Path('./models/transformer')

# Define the file paths
model_path = base_dir / 'model.pkl'
vocab_transform_path = base_dir / 'vocab_transform.pkl'
token_transform_path = base_dir / 'token_transform.pkl'

# Load the model, vocab_transform, and token_transform
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vocab_transform_path, 'rb') as model_file:
    vocab_transform = pickle.load(model_file)

with open(token_transform_path, 'rb') as model_file:
    token_transform = pickle.load(model_file)

# Create the Flask app
app = Flask(__name__, static_folder='./static', static_url_path='')

# Enable CORS
CORS(app)

# Define the routes
@app.route('/')
def index_page():
    return app.send_static_file('index.html')

@app.route('/<path:path>')
def serve_custom_path(path):
    return send_from_directory('./', path)

@app.route('/predict', methods=['GET'])
def predict_price():
    input_search_text = request.args.get('search')
    text_transform = get_text_transform(token_transform, vocab_transform)
    prediction = perform_sample_translation(model, input_search_text, text_transform, vocab_transform)

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)