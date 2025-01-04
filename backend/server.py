from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import io

# Initialize the Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# Load the trained model
model = tf.keras.models.load_model('pox_classifier_model.h5')

# Define the class labels
class_labels = ['chickenpox', 'cowpox', 'healthy', 'monkeypox']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Convert the file to a BytesIO object and process it
        img = image.load_img(io.BytesIO(file.read()), target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = class_labels[np.argmax(prediction)]

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/train', methods=['POST'])
def train():
    return jsonify({'status': 'Training not implemented in this example'})


if __name__ == '__main__':
    app.run(debug=True)
