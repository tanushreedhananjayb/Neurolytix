🧠 Neurolytix
Smart MRI Tumor Detection System

* Overview :
Neurolytix is an intelligent MRI scan analysis web application designed to assist in the early detection of brain tumors. It provides a user-friendly platform for uploading brain MRI images, analyzes them using deep learning-based feature extraction, and classifies the tumor type with a reliable machine learning model. The results are presented in a medically themed, professionally styled report that mimics the feel of a diagnostic prescription.

This tool can be particularly helpful in aiding radiologists and healthcare practitioners by providing a quick, second opinion based on pre-trained data and models.

* Objective :
The main aim of Neurolytix is to simplify and streamline the detection of brain tumors through automation. By combining the power of convolutional neural networks (CNN) for feature extraction with the speed and accuracy of a Random Forest Classifier, the system delivers predictions that are both efficient and interpretable.

* Key Features :
- Upload and analyze brain MRI scans instantly.

* Detects four conditions:
→ No Tumor
→ Glioma Tumor
→ Meningioma Tumor
→ Pituitary Tumor

- Uses MobileNetV2 CNN architecture for high-quality feature extraction.

- Random Forest Classifier for classification based on extracted image features.

- Medical-style user interface with dark theme and professional report layout.

- Responsive and creative UI built with Bootstrap 5 and custom CSS.

- Deployable securely with Render for public access.

* Technologies Used
-> Frontend:
  - HTML5, CSS3
  - Bootstrap 5 (custom themed) 
  - Responsive design principles
    
-> Backend:
  - Python 3.10
  - Flask Web Framework

-> Machine Learning:
  - TensorFlow, Keras (MobileNetV2)
  - Scikit-learn (Random Forest Classifier)
  - OpenCV for image preprocessing

-> Others:
  - Joblib for model serialization
  - Render for deployment
  - Git & GitHub for version control

* How It Works :

1. A user uploads a brain MRI image via the web interface.
2. The image is resized and preprocessed, then passed through a MobileNetV2 CNN model to extract features.
3. These features are classified using a trained Random Forest model.
4. The system outputs the tumor type (if any) in a clear, styled diagnostic report format.
5. Users receive messages like “Congrats! No tumor detected.” or recommendations for specific tumor types.

* Project Structure :

  Neurolytix/
│
├── app.py                            # Main Flask server
├── tuned_random_forest_model.pkl    # Trained ML model
├── requirements.txt                 # Python dependencies
├── static/
│   ├── uploads/                     # Uploaded MRI scans
│   └── images/                      # Logo, favicon
├── templates/
│   ├── index.html                   # Upload page
│   └── result.html                  # Result report
├── README.md                        # Project description

* Getting Started (Local Setup):

1. Clone the repository:

git clone https://github.com/tanushreedhananjayb/Neurolytix.git
cd Neurolytix

2. Create a virtual environment and activate it (recommended):

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies:

pip install -r requirements.txt

4. Run the application:

python app.py

* Contact :
Tanushree Dhananjay Bhamare
🔗 GitHub: tanushreedhananjayb
🔗 LinkedIn: https://www.linkedin.com/in/tanushree-dhananjay-bhamare-9219b724b/
