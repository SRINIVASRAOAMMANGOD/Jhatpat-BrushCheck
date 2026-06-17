from flask import Flask, render_template, request, redirect
import os

# Reduce TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from PIL import Image
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
import base64
import re
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'model', 'keras_model.h5')
labels_path = os.path.join(base_dir, 'model', 'labels.txt')

# Lazy-load model
model = None

def get_model():
    global model

    if model is None:
        print("Loading model...")
        model = load_model(model_path, compile=False)

    return model

# Load labels once
with open(labels_path, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

bacteria_counts = {
    "new": "Low",
    "used": "Medium",
    "dirty": "High",
    "frayed": "Very High"
}

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(e):
    return "File too large. Max 50MB allowed.", 413

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_and_predict', methods=['POST'])
def upload_and_predict():

    filename = None

    captured_image = request.form.get('captured_image')

    if captured_image:
        try:
            img_str = re.search(r'base64,(.*)', captured_image).group(1)
            img_data = base64.b64decode(img_str)

            filename = "captured.png"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            with open(filepath, 'wb') as f:
                f.write(img_data)

        except Exception as e:
            return f"Failed to decode captured image: {e}", 400

    elif 'image' in request.files:

        file = request.files['image']

        if file.filename == '':
            return redirect('/')

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        try:
            file.save(filepath)
        except Exception as e:
            return f"Failed to save uploaded file: {e}", 500

    else:
        return "No image provided", 400

    try:
        image = Image.open(filepath).convert('RGB')
        image = image.resize((224, 224))

        image_array = np.asarray(image, dtype=np.float32)
        image_array = image_array.reshape(1, 224, 224, 3)

        image_array = (image_array / 127.5) - 1

        prediction = get_model().predict(image_array, verbose=0)

        index = np.argmax(prediction)

        class_name = class_names[index]
        confidence = float(prediction[0][index] * 100)

        bacteria = bacteria_counts.get(
            class_name.lower(),
            "Unknown"
        )

    except Exception as e:
        return f"Prediction error: {e}", 500

    return render_template(
        'result.html',
        filename=filename,
        prediction=class_name,
        confidence=f"{confidence:.2f}",
        bacteria=bacteria
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))

    app.run(
        host='0.0.0.0',
        port=port,
        debug=False
    )
