import os
import cv2
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model

# Dataset path (update if necessary)
dataset_path = "D:\\Labmentix_Braintumor Project\\Tumour-20250723T125127Z-1-001\\Tumour\\train"
labels_dict = {
    "no": 0,
    "no_tumor": 0,
    "glioma": 1,
    "meningioma": 2,
    "pituitary": 3
}

valid_extensions = ('.jpg', '.jpeg', '.png')

# Load MobileNetV2 for CNN feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
model_cnn = Model(inputs=base_model.input, outputs=base_model.output)

# Step 1: Extract features
print("[INFO] Extracting features...")
data = []
labels = []

for label in os.listdir(dataset_path):
    label_path = os.path.join(dataset_path, label)
    if not os.path.isdir(label_path):
        continue

    folder_key = label.lower().replace(" ", "_").replace("-", "_")

    if folder_key not in labels_dict:
        print(f"[WARNING] Skipping unknown label folder: {label}")
        continue

    for img_name in os.listdir(label_path):
        if not img_name.lower().endswith(valid_extensions):
            continue

        img_path = os.path.join(label_path, img_name)
        try:
            image = load_img(img_path, target_size=(128, 128))
            image = img_to_array(image)
            image = preprocess_input(image)
            image = np.expand_dims(image, axis=0)

            features = model_cnn.predict(image, verbose=0)
            flattened = features.flatten()

            data.append(flattened)
            labels.append(labels_dict[folder_key])

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path}: {e}")

print(f"[INFO] Total processed samples: {len(data)} features, {len(labels)} labels")

# Step 2: Train/Test Split
X = np.array(data)
y = np.array(labels)

if len(X) != len(y):
    print("[ERROR] Mismatch between features and labels. Aborting.")
    exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 3: Train Random Forest Classifier
print("[INFO] Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 4: Evaluate the model
y_pred = rf.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Step 5: Save the model
model_output_path = "tuned_random_forest_model.pkl"
joblib.dump(rf, model_output_path)
print(f"[INFO] Model saved to '{model_output_path}'")
