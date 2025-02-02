from flask import Flask, request, send_from_directory, jsonify
from flask_cors import CORS
import pickle
import dill

from helper.translate import perform_sample_translation

from classes.transformer.decoder import Decoder
from classes.transformer.encoder import Encoder
from classes.transformer.encoder_layer import EncoderLayer
from classes.transformer.decoder_layer import DecoderLayer
from classes.transformer.multi_head_attention_layer import MultiHeadAttention
from classes.transformer.position_wise_feed_forward_layer import PositionwiseFeedforward
from classes.transformer.sequence_to_sequence_transformer import SequenceToSequenceTransformer

# Load the model and the scaler
with open('./models/transformer/model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('./models/transformer/vocab_transform.pkl', 'rb') as model_file:
    vocab_transform = pickle.load(model_file)

with open('../notebooks/helper/transformer/text_transform.pkl', 'rb') as model_file:
    text_transform = dill.load(model_file)

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

    prediction = perform_sample_translation(model, input_search_text, text_transform, vocab_transform)

    return jsonify(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)