import os
import random
import shutil

# Func buat mindahin keseluruhan dataset
def move_sample_dataset(src_dir, dest_dir, sample_size=50000):
    # List semua kelas (folder) di src_dir
    classes = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    # Buat folder kelas di dest_dir
    for cls in classes:
        os.makedirs(os.path.join(dest_dir, cls), exist_ok=True)

    # Kumpulin semua file dengan label kelasnya
    all_files = []
    for cls in classes:
        cls_path = os.path.join(src_dir, cls)
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if os.path.isfile(os.path.join(cls_path, f))]
        all_files.extend([(f, cls) for f in files])

    # Cek kalau sample_size lebih besar dari total data
    if sample_size > len(all_files):
        print(f"Sample size {sample_size} lebih besar dari total data {len(all_files)}. Menggunakan semua data.")
        sample_files = all_files
    else:
        sample_files = random.sample(all_files, sample_size)

    print(f"Moving {len(sample_files)} files dari total {len(all_files)} files.")

    # Move file ke folder tujuan
    for filepath, cls in sample_files:
        dest_path = os.path.join(dest_dir, cls, os.path.basename(filepath))
        shutil.move(filepath, dest_path)

    print(f"Selesai move data subset ke {dest_dir}.")

# Pakai fungsi ini BUAT MINDAHIN DATASET
# src = "data/train"
# dest = "data/train_subset_50k"
# move_sample_dataset(src, dest, sample_size=50000)

# Func buat mengelompokkan otomatis data (ex. test)
source_dir = "data/test"
target_dir = "data/test_sorted"

# Bikin folder target kalau belum ada
os.makedirs(target_dir, exist_ok=True)

# Mapping kasar manual dari pola nama ke folder
# Disesuaikan dengan dataset-mu
label_map = {
    "AppleCedarRust": "apple_cedar_apple_rust",
    "AppleScab": "apple_apple_scab",
    "CherryHealthy": "cherry_healthy",
    "CherryPowderyMildew": "cherry_powdery_mildew",
    "CornCommonRust": "corn_common_rust",
    "CornHealthy": "corn_healthy",
    "GrapeBlackRot": "grape_black_rot",
    "GrapeEsca": "grape_esca",
    "GrapeHealthy": "grape_healthy",
    "PeachHealthy": "peach_healthy",
    "PeachBacterialSpot": "peach_bacterial_spot",
    "PepperBacterialSpot": "pepper_bacterial_spot",
    "PepperHealthy": "pepper_healthy",
    "PotatoEarlyBlight": "potato_early_blight",
    "PotatoHealthy": "potato_healthy",
    "RiceBlight": "rice_leaf_bacterial_blight",
    "RiceBrownSpot": "rice_leaf_brown_spot",
    "RiceSmut": "rice_leaf_smut",
    "StrawberryHealthy": "strawberry_healthy",
    "StrawberryLeafScorch": "strawberry_leaf_scorch",
    "TomatoEarlyBlight": "tomato_early_blight",
    "TomatoHealthy": "tomato_healthy",
}

# Proses semua file
for filename in os.listdir(source_dir):
    filepath = os.path.join(source_dir, filename)

    if os.path.isfile(filepath):
        for key, label in label_map.items():
            if filename.startswith(key):
                label_folder = os.path.join(target_dir, label)
                os.makedirs(label_folder, exist_ok=True)

                new_path = os.path.join(label_folder, filename)
                shutil.move(filepath, new_path)
                print(f"Moved {filename} → {label}/")
                break


