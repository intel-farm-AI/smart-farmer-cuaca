import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

CFG = {
    "train_dir": "data/train/",
    "model_path": "models/compiled/plant_disease_model_v2.h5",
    "label_path": "models/labels.json",
    "history_path": "models/history.json",
    "img_size": (224, 224),
    "batch_size": 16,
    "epochs": 8,
    "lr": 1e-4,
    "patience": 5,
    "min_lr": 1e-7,
    "fine_tune_at": 100,  # layer index di base_model dari mana unfreeze (optional)
    "fine_tune_epochs": 3,
    "fine_tune_lr": 1e-5,
}

def get_data_generators(directory, img_size, batch_size):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=30,
        zoom_range=0.3,
        horizontal_flip=True,
    )
    train_gen = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True,
    )
    val_gen = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
    )
    return train_gen, val_gen

def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=CFG["img_size"] + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # freeze awal

    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model

def adapt_model(old_model, new_num_classes):
    print("🔧 Adaptasi model: ganti output layer")

    # freeze semua kecuali output layer nanti dibuat baru
    for layer in old_model.layers:
        layer.trainable = False

    # Ambil base_model dari old_model input sampai layer sebelum output terakhir
    base_model = old_model.get_layer(index=0)  # MobileNetV2 biasanya layer ke-0 di Sequential atau Functional
    
    # Buat ulang head baru dengan jumlah kelas baru
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    new_output = layers.Dense(new_num_classes, activation='softmax', name='predictions')(x)

    new_model = models.Model(inputs=base_model.input, outputs=new_output)
    return new_model

def load_existing_model():
    if os.path.exists(CFG["model_path"]):
        print(f"📦 Memuat model lama dari {CFG['model_path']}")
        return load_model(CFG["model_path"])
    return None

def unfreeze_model(model, fine_tune_at):
    print(f"🔓 Unfreeze model mulai layer ke-{fine_tune_at} untuk fine-tuning")
    base_model = model.layers[0]  # asumsi base_model di index 0
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=CFG['patience'], restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=CFG['min_lr']),
        ModelCheckpoint(CFG['model_path'], monitor='val_loss', save_best_only=True)
    ]

def train():
    print("✅ GPU available:", tf.config.list_physical_devices('GPU'))

    train_gen, val_gen = get_data_generators(CFG["train_dir"], CFG["img_size"], CFG["batch_size"])
    num_classes = train_gen.num_classes
    print(f"✅ Terdeteksi {num_classes} kelas.")

    with open(CFG["label_path"], "w") as f:
        json.dump(train_gen.class_indices, f)
        print(f"📝 Label disimpan ke {CFG['label_path']}")

    model = None
    old_model = load_existing_model()
    if old_model:
        old_num_classes = old_model.output_shape[-1]
        if old_num_classes != num_classes:
            print(f"⚠️ Jumlah kelas berubah ({old_num_classes} → {num_classes}), adaptasi model...")
            model = adapt_model(old_model, num_classes)
        else:
            model = old_model
            print("📦 Menggunakan model lama dengan jumlah kelas yang sama.")
    else:
        print("✨ Membuat model baru...")
        model = build_model(num_classes)

    # Compile dengan learning rate awal
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CFG["lr"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Training fase 1: train output layer (base model frozen)
    print("🚀 Training fase 1: training output layer dengan base model dibekukan")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CFG["epochs"]//2,
        callbacks=get_callbacks()
    )

    # Training fase 2: fine-tune sebagian base model (optional)
    unfreeze_model(model, CFG['fine_tune_at'])

    # Compile ulang dengan learning rate lebih kecil untuk fine-tune
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CFG["lr"] / 10),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("🚀 Training fase 2: fine-tuning sebagian base model")
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CFG["epochs"] - (CFG["epochs"] // 2),
        callbacks=get_callbacks()
    )

    # Gabungkan history
    for key in history.history:
        history.history[key].extend(history_fine.history[key])

    # Simpan history training
    with open(CFG["history_path"], "w") as f:
        json.dump(history.history, f)
        print(f"📊 History training disimpan ke {CFG['history_path']}")

    print("✅ Training selesai.")
    
def resume_train():
    if not os.path.exists(CFG["model_path"]):
        print("❌ Model belum dilatih. Jalankan train() dulu.")
        return

    print("📦 Melanjutkan training dari model lama...")
    model = load_model(CFG["model_path"])
    
    # Unfreeze base_model jika ingin fine-tune full
    if isinstance(model.layers[0], tf.keras.Model):
        print("🔓 Unfreezing base model for fine-tuning...")
        model.layers[0].trainable = True

    # Recompile dengan learning rate kecil
    model.compile(
        optimizer=optimizers.Adam(learning_rate=CFG["fine_tune_lr"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    train_gen, val_gen = get_data_generators(CFG["train_dir"], CFG["img_size"], CFG["batch_size"])
    
    print(f"🚀 Resume training for {CFG['fine_tune_epochs']} epochs...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CFG["fine_tune_epochs"],
        callbacks=get_callbacks()
    )

    # Update history
    if os.path.exists(CFG["history_path"]):
        with open(CFG["history_path"], "r") as f:
            old_history = json.load(f)
        for key in history.history:
            old_history[key] += history.history[key]
    else:
        old_history = history.history

    with open(CFG["history_path"], "w") as f:
        json.dump(old_history, f)
        print(f"📊 History diperbarui di {CFG['history_path']}")

    print("✅ Fine-tune selesai.")


if __name__ == "__main__":
    # train() # kalau mau dari awal
    resume_train() # kalau lanjutin dari model sebelumnya
