from flask import Flask, render_template, request, redirect, url_for
import os
from PIL import Image
import numpy as np
from keras.models import load_model
from werkzeug.utils import secure_filename
import base64
import re
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max upload size

# Make sure upload folder exists at startup
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model and class names once globally, stripping whitespace
model = load_model('model/keras_model.h5', compile=False)
with open('model/labels.txt', 'r') as f:
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
    captured_image = request.form.get('captured_image')

    if captured_image:
        # Handle base64 webcam image
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
            return redirect(request.url)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        try:
            file.save(filepath)
        except Exception as e:
            return f"Failed to save uploaded file: {e}", 500
    else:
        return "No image provided", 400

    # Process image and predict
    try:
        image = Image.open(filepath).convert('RGB').resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1  # normalize to [-1,1]

        prediction = model.predict(image_array)
        index = np.argmax(prediction)
        class_name = class_names[index].strip()  # Strip whitespace here too
        confidence = prediction[0][index] * 100

        bacteria = bacteria_counts.get(class_name.lower(), "Unknown")
    except Exception as e:
        return f"Prediction error: {e}", 500

    return render_template('result.html',
                           filename=filename,
                           prediction=class_name,
                           confidence=f"{confidence:.2f}",
                           bacteria=bacteria)

if __name__ == '__main__':
    app.run(debug=True)
