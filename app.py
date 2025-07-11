import os
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS

# ===== Inisialisasi Flask App =====
app = Flask(__name__)
CORS(app)

# ===== Load model hanya sekali saat startup =====
MODEL_PATH = "models/compiled/plant_disease_model_v2.keras"
model = load_model(MODEL_PATH)

# Ukuran input gambar sesuai training
IMG_SIZE = (224, 224)

# ===== Label (urutannya harus sama kayak training) =====
# Optional: bisa juga load dari labels.json
LABELS = "models/labels.json"

# ===== Preprocess image sebelum masuk model =====
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)  # tambahkan batch dimensi
    return image

# ===== Endpoint untuk prediksi =====
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['file']
    try:
        image = Image.open(file).convert("RGB")
        input_tensor = preprocess_image(image)
        predictions = model.predict(input_tensor)[0]

        top_index = np.argmax(predictions)
        confidence = float(predictions[top_index])
        label = LABELS[top_index]

        return jsonify({
            "class": label,
            "confidence": round(confidence, 4)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ====== Run server di localhost =====
if __name__ == '__main__':
    app.run(debug=True, port=5000)
