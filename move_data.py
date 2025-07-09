import os
import random
import shutil

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

# Pakai fungsi ini
src = "data/train"
dest = "data/train_subset_50k"
move_sample_dataset(src, dest, sample_size=50000)
