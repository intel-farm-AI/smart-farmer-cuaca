import numpy as np
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load Model
model_path = "models/compiled/plant_disease_model_v2.h5"
model = load_model(model_path)
print(f"✅ Model Loaded: {model_path}")

# Load label dari file training
with open("models/labels.json") as f:
    class_indices = json.load(f)

# Susun list labels sesuai urutan index (biar akurat)
labels = [None] * len(class_indices)
for name, idx in class_indices.items():
    labels[idx] = name

# Load dataset uji
test_dir = "data/test/"
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=16,
    class_mode=None,     # penting: karena kita gak punya ground truth
    shuffle=True
)

print(f"🔍 Begin Testing on {len(test_gen.filenames)} images...")

# Predict
preds = model.predict(test_gen)
predicted_class_indices = np.argmax(preds, axis=1)
predicted_labels = [labels[i] for i in predicted_class_indices]

# Print hasil prediksi
for fname, label in zip(test_gen.filenames, predicted_labels):
    print(f"{fname} => {label}")
