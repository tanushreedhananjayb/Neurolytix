import os
import sys
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Suppress TensorFlow unnecessary logs and GPU access
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')  # Avoid fake GPU usage

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load Random Forest model (expects 20480 features)
model = joblib.load('tuned_random_forest_model.pkl')

# MobileNetV2 feature extractor (no pooling, gives 20480 features)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model_cnn = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.output)

# Label Map and Message
label_map = {
    0: "No Tumor",
    1: "Glioma Tumor",
    2: "Meningioma Tumor",
    3: "Pituitary Tumor"
}

message_map = {
    0: "🟢 Congratulations! No tumor detected. Stay healthy and positive!",
    1: "⚠️ Glioma tumor detected. Please consult a neurologist immediately for further examination.",
    2: "⚠️ Meningioma tumor detected. Medical attention is strongly advised.",
    3: "⚠️ Pituitary tumor detected. Schedule a clinical evaluation as soon as possible."
}

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing
def preprocess_image(image_path):
    image = load_img(image_path, target_size=(128, 128))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)

    with tf.device('/CPU:0'):
        features = model_cnn.predict(image, verbose=0)  # Output shape: (1, 4, 4, 1280)

    features_flat = features.flatten().reshape(1, -1)  # Shape: (1, 20480)
    return features_flat

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(url_for('home'))

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            features = preprocess_image(filepath)
            prediction = model.predict(features)[0]
            predicted_label = label_map.get(prediction, "Unknown")
            diagnosis_message = message_map.get(prediction, "Unknown diagnosis. Please retry.")

            return render_template('result.html', prediction=predicted_label, message=diagnosis_message, img_path=filepath)
        except Exception as e:
            print(f"Prediction error: {e}", file=sys.stderr)
            return "Error during prediction. Please try again later.", 500

    return redirect(url_for('home'))

# Entrypoint
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    try:
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Error starting Flask app: {e}", file=sys.stderr)
