import os
import json
import datetime
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# ===== Konfigurasi Utama (mudah disesuaikan) =====
CFG = {
    "train_dir": "data/train/",  # direktori data training
    "model_path": "models/compiled/plant_disease_model_v2.keras",  # path untuk menyimpan/memuat model
    "label_path": "models/labels.json",  # path untuk menyimpan label klasifikasi
    "history_path": "models/history.json",  # path untuk menyimpan history training
    "img_size": (224, 224),
    "batch_size": 16,
    "epochs": 10,
    "lr": 1e-4,
    "patience": 5,
    "min_lr": 1e-7,
    "fine_tune_at": 50,  # layer index untuk mulai unfreeze saat fine-tuning
    "fine_tune_epochs": 3,
    "fine_tune_lr": 1e-7,
}

# ===== Membuat data generator untuk training & validasi =====
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

# ===== Membangun model dari awal (MobileNetV2 + dense layer custom) =====
def build_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=CFG["img_size"] + (3,),
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False  # freeze base model

    # Tambahkan top model
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = models.Model(inputs=base_model.input, outputs=output)
    return model

# ===== Adaptasi model jika jumlah kelas berubah =====
def adapt_model(old_model, new_num_classes):
    print("🔧 Adaptasi model: ganti output layer")
    for layer in old_model.layers:
        layer.trainable = False

    # Temukan layer terakhir yang bukan dense sebagai base_model
    for i in range(len(old_model.layers) - 1, -1, -1):
        if not isinstance(old_model.layers[i], layers.Dense):
            base_output = old_model.layers[i].output
            break

    x = layers.Dense(128, activation='relu')(base_output)
    x = layers.Dropout(0.3)(x)
    new_output = layers.Dense(new_num_classes, activation='softmax', name='predictions')(x)

    return models.Model(inputs=old_model.input, outputs=new_output)

# ===== Cek dan load model jika tersedia =====
def load_existing_model():
    if os.path.exists(CFG["model_path"]):
        print(f"📦 Memuat model lama dari {CFG['model_path']}")
        return load_model(CFG["model_path"])
    return None

# ===== Fungsi unfreeze base model sebagian untuk fine-tuning =====
def unfreeze_model(model, fine_tune_at):
    print(f"🔓 Unfreeze model mulai layer ke-{fine_tune_at} untuk fine-tuning")

    # Temukan semua layer yang termasuk bagian MobileNetV2 (berdasarkan pola nama)
    mobilenetv2_layers = [layer for layer in model.layers if "Conv1" in layer.name or "block" in layer.name or "expanded_conv" in layer.name]

    if not mobilenetv2_layers:
        raise ValueError("❌ Tidak bisa menemukan layer-layer MobileNetV2 untuk fine-tuning.")

    # Unfreeze semua layer MobileNetV2
    for i, layer in enumerate(mobilenetv2_layers):
        layer.trainable = i >= fine_tune_at

    print(f"✅ Total {len(mobilenetv2_layers)} layer dari MobileNetV2 ditemukan. {len(mobilenetv2_layers) - fine_tune_at} di-unfreeze.")


# ===== Callback standar untuk training =====
def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=CFG['patience'], restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=CFG['min_lr']),
        ModelCheckpoint(CFG['model_path'], monitor='val_loss', save_best_only=True)
    ]

# ===== Fungsi utama training model =====
def train():
    print("✅ GPU available:", tf.config.list_physical_devices('GPU'))

    train_gen, val_gen = get_data_generators(CFG["train_dir"], CFG["img_size"], CFG["batch_size"])
    num_classes = train_gen.num_classes
    print(f"✅ Terdeteksi {num_classes} kelas.")

    with open(CFG["label_path"], "w") as f:
        json.dump(train_gen.class_indices, f)
        print(f"📝 Label disimpan ke {CFG['label_path']}")

    model = load_existing_model()
    if model:
        old_num_classes = model.output_shape[-1]
        if old_num_classes != num_classes:
            print(f"⚠️ Jumlah kelas berubah ({old_num_classes} → {num_classes}), adaptasi model...")
            model = adapt_model(model, num_classes)
        else:
            print("📦 Menggunakan model lama dengan jumlah kelas yang sama.")
    else:
        print("✨ Membuat model baru...")
        model = build_model(num_classes)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=CFG["lr"]),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("🚀 Training fase 1: training output layer dengan base model dibekukan")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CFG["epochs"] // 2,
        callbacks=get_callbacks()
    )

    unfreeze_model(model, CFG['fine_tune_at'])

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

    for key in history.history:
        history.history[key].extend(history_fine.history[key])

    with open(CFG["history_path"], "w") as f:
        json.dump(history.history, f)
        print(f"📊 History training disimpan ke {CFG['history_path']}")

    model.save(CFG["model_path"], save_format="keras")
    print("✅ Training selesai.")

# ===== Fungsi lanjutan untuk melanjutkan training =====
def resume_train():
    if not os.path.exists(CFG["model_path"]):
        print("❌ Model belum dilatih. Jalankan train() dulu.")
        return

    print("📦 Melanjutkan training dari model lama...")
    model = load_model(CFG["model_path"])

    unfreeze_model(model, CFG['fine_tune_at'])

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

    model.save(CFG["model_path"], save_format="keras")
    print("✅ Fine-tune selesai.")

if __name__ == "__main__":
    train()
    # resume_train()
