import os, sys
import joblib
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.config.set_visible_devices([], 'GPU')  # Avoid allocating GPU memory (even if it's fake in Railway)
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Silence TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load Models
try:
    model = joblib.load('tuned_random_forest_model.pkl')
    model_cnn = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(128, 128, 3))
except Exception as e:
    print(f"[MODEL LOAD ERROR] {e}", file=sys.stderr)
    raise

label_map = {
    0: "No Tumor",
    1: "Glioma Tumor",
    2: "Meningioma Tumor",
    3: "Pituitary Tumor"
}

message_map = {
    0: "🟢 No tumor detected. Stay healthy!",
    1: "⚠️ Glioma tumor detected. Please consult a neurologist.",
    2: "⚠️ Meningioma tumor detected. Medical attention recommended.",
    3: "⚠️ Pituitary tumor detected. Schedule a medical evaluation."
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    try:
        image = load_img(image_path, target_size=(128, 128))
        image = img_to_array(image)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)

        with tf.device('/CPU:0'):
            features = model_cnn.predict(image, verbose=0)
        return features.reshape(1, -1)
    except Exception as e:
        print(f"[IMAGE PREPROCESS ERROR] {e}", file=sys.stderr)
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            print("[ERROR] No file part in request", file=sys.stderr)
            return redirect(url_for('home'))

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            features = preprocess_image(filepath)
            if features is None:
                raise ValueError("Feature extraction failed.")

            prediction = model.predict(features)[0]
            label = label_map.get(prediction, "Unknown")
            message = message_map.get(prediction, "Please try again.")

            return render_template('result.html', prediction=label, message=message, img_path=filepath)

        else:
            print("[ERROR] Invalid file format", file=sys.stderr)
            return redirect(url_for('home'))

    except Exception as e:
        print(f"[PREDICT ROUTE ERROR] {e}", file=sys.stderr)
        return render_template('error.html', error_message="Something went wrong. Please try again.")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    try:
        app.run(host="0.0.0.0", port=port)
    except Exception as e:
        print(f"Error starting Flask app: {e}", file=sys.stderr)
