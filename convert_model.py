import tensorflow as tf

# Load dari .h5
model = tf.keras.models.load_model("plant_disease_model.h5", compile=False)

# Simpan ulang ke format SavedModel
model.export("plant_disease_model")

print("✅ Model berhasil dikonversi ke SavedModel format!")
