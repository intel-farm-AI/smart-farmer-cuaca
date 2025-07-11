import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ===== CONFIG (adaptif dari train_model.py) =====
CFG = {
    "model_path": "models/compiled/plant_disease_model_v2.keras",
    "label_path": "models/labels.json",
    "img_size": (224, 224),
    "batch_size": 16,
    "test_dir": "data/test/"
}

# ===== LOAD MODEL =====
if not os.path.exists(CFG["model_path"]):
    raise FileNotFoundError(f"Model not found at: {CFG['model_path']}")
model = load_model(CFG["model_path"])
print(f"✅ Model Loaded: {CFG['model_path']}")

# ===== LOAD LABELS =====
if not os.path.exists(CFG["label_path"]):
    raise FileNotFoundError(f"Label file not found at: {CFG['label_path']}")

with open(CFG["label_path"]) as f:
    class_indices = json.load(f)

# Susun label berdasarkan index
labels = [None] * len(class_indices)
for name, idx in class_indices.items():
    labels[idx] = name

# ===== LOAD TEST DATASET =====
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    CFG["test_dir"],
    target_size=CFG["img_size"],
    batch_size=CFG["batch_size"],
    class_mode='categorical',
    shuffle=False
)

print(f"🔍 Begin Testing on {len(test_gen.filenames)} images...")

# ===== PREDICT =====
preds = model.predict(test_gen)
predicted_class_indices = np.argmax(preds, axis=1)
true_class_indices = test_gen.classes
predicted_labels = [labels[i] for i in predicted_class_indices]
true_labels = [labels[i] for i in true_class_indices]

# ===== OUTPUT RESULT =====
print("\n===== Classification Report =====")
print(classification_report(true_class_indices, predicted_class_indices, target_names=labels))

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(true_class_indices, predicted_class_indices)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ===== OPTIONAL: Per-Image Output =====
print("\n===== Sample Predictions =====")
for fname, pred, true in zip(test_gen.filenames, predicted_labels, true_labels):
    print(f"{fname} => predicted: {pred} | actual: {true}")
