import tensorflowjs as tfjs
import tensorflow as tf
import os

# === Konfigurasi path ===
KERAS_MODEL_PATH = "models/compiled/plant_disease_model_v2.keras"  # Ganti sesuai lokasi model
TFJS_OUTPUT_DIR = "models/web_model/"  # Folder output TF.js

# === Muat model dari .keras file ===
print("📦 Memuat model dari:", KERAS_MODEL_PATH)
model = tf.keras.models.load_model(KERAS_MODEL_PATH)

# === Pastikan direktori output ada ===
os.makedirs(TFJS_OUTPUT_DIR, exist_ok=True)

# === Konversi model ke TensorFlow.js format ===
print("🔄 Mengonversi model ke TensorFlow.js format...")
tfjs.converters.save_keras_model(model, TFJS_OUTPUT_DIR)

print(f"✅ Konversi selesai! Model TF.js disimpan di: {TFJS_OUTPUT_DIR}")
print("📁 Termasuk: model.json dan file .bin (berisi bobot model)")
