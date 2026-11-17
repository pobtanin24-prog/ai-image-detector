import os
import random
import shutil

# ตำแหน่ง dataset ตัวเต็ม
SOURCE = "cifake-full"

# ตำแหน่ง dataset ขนาดเล็กที่เราจะสร้าง
DESTINATION = "data"

# จำนวนภาพที่ต้องการ
TRAIN_PER_CLASS = 2000
TEST_PER_CLASS = 500

# ทำให้สุ่มเหมือนเดิมทุกครั้ง
random.seed(42)


def copy_random_images(source_folder, destination_folder, amount):
    os.makedirs(destination_folder, exist_ok=True)

    files = os.listdir(source_folder)
    selected_files = random.sample(files, amount)

    for filename in selected_files:
        source = os.path.join(source_folder, filename)
        destination = os.path.join(destination_folder, filename)

        shutil.copy2(source, destination)


# Train
copy_random_images(
    "cifake-full/train/REAL",
    "data/train/REAL",
    TRAIN_PER_CLASS
)

copy_random_images(
    "cifake-full/train/FAKE",
    "data/train/FAKE",
    TRAIN_PER_CLASS
)

# Test
copy_random_images(
    "cifake-full/test/REAL",
    "data/test/REAL",
    TEST_PER_CLASS
)

copy_random_images(
    "cifake-full/test/FAKE",
    "data/test/FAKE",
    TEST_PER_CLASS
)

print("Dataset prepared!")
print("Train: 4000 images")
print("Test: 1000 images")