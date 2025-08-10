import os
import numpy as np
from keras.models import load_model
from PIL import Image
import re

# Base directory of the project
BASE_DIR = os.path.dirname(__file__)

# Paths (relative, so they work on Railway)
model_path = os.path.join(BASE_DIR, "model", "keras_model.h5")
labels_path = os.path.join(BASE_DIR, "model", "labels.txt")
images_folder = os.path.join(BASE_DIR, "images")

# Load model and labels
model = load_model(model_path, compile=False)
class_names = open(labels_path, "r").read().splitlines()

# Simulated bacteria count mapping
BACTERIA_MAPPING = {
    "new": 5,
    "used": 200,
    "dirty": 500,
    "frayed": 1000
}

print("Class indices: 0 - new, 1 - used, 2 - dirty, 3 - frayed\n")

# Loop through all images in folder
for img_name in os.listdir(images_folder):
    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        img_path = os.path.join(images_folder, img_name)

        # Load and preprocess image
        image = Image.open(img_path).convert("RGB")
        image = image.resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_array = (image_array / 127.5) - 1

        # Predict
        prediction = model.predict(image_array, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index].lower()
        class_name = re.sub(r"^\d+\s*", "", class_name)  # Remove leading numbers
        confidence_score = prediction[0][index]

        # Get bacteria count from mapping
        breezel_count = BACTERIA_MAPPING.get(class_name, "Unknown")

        print(f"{img_name} --> {class_name} ({confidence_score*100:.2f}%) | Breezel (bacteria count): {breezel_count}")
