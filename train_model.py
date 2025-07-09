import os
import tensorflow as tf
import datetime
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===== CONFIGURATION =====
train_dir = "data/train_subset_50k/"
model_path = "models/plant_disease_model_50K.h5"
img_size = (224, 224)
batch_size = 16
epochs = 30
lr = 1e-5
patience = 5

# ===== CEK GPU =====
print("✅ GPU available:", tf.config.list_physical_devices('GPU'))

# ===== DATA LOADER =====
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
)

train_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_gen = datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = train_gen.num_classes
print(f"✅ Kelas terdeteksi: {num_classes} kelas")

# ===== MODEL BUILDER =====
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True  # Fine-tune semua layer

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# ===== CALLBACKS =====
callbacks = [
    EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7),
    ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
]

# ===== LOAD / BUILD MODEL =====
if os.path.exists(model_path):
    model = load_model(model_path)
    print(f"📦 Model diload dari: {model_path}")
else:
    model = build_model(num_classes)
    print("✨ Model baru dibuat.")

model.compile(
    optimizer=optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=['accuracy'],
)

# ===== TRAINING =====
start_time = datetime.datetime.now()
print(f"🚀 Training dimulai: {start_time.strftime('%H:%M:%S')}")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs,
    callbacks=callbacks
)

end_time = datetime.datetime.now()
print(f"✅ Training selesai: {end_time.strftime('%H:%M:%S')} (durasi: {str(end_time - start_time).split('.')[0]})")
