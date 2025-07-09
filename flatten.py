import os
import shutil

def flatten_dataset(root_dir):
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(root_dir, split)
        if not os.path.exists(split_path):
            continue

        for plant_folder in os.listdir(split_path):
            plant_path = os.path.join(split_path, plant_folder)
            if not os.path.isdir(plant_path):
                continue

            for disease_folder in os.listdir(plant_path):
                disease_path = os.path.join(plant_path, disease_folder)
                if not os.path.isdir(disease_path):
                    continue

                # Format nama folder baru: tanaman_penyakit
                clean_plant = plant_folder.lower().replace(" ", "_")
                clean_disease = disease_folder.lower().replace(" ", "_")
                new_class_name = f"{clean_plant}_{clean_disease}"
                new_class_path = os.path.join(split_path, new_class_name)

                os.makedirs(new_class_path, exist_ok=True)

                # Pindahkan semua gambar
                for filename in os.listdir(disease_path):
                    src = os.path.join(disease_path, filename)
                    dst = os.path.join(new_class_path, filename)
                    shutil.move(src, dst)

            # Hapus folder tanaman setelah semua file dipindah
            shutil.rmtree(plant_path)

        print(f"✅ Beres untuk folder '{split}'")

# 🔧 Ganti path di bawah sesuai lokasi dataset kamu
dataset_root = r"../data_smartfarm/"
flatten_dataset(dataset_root)
