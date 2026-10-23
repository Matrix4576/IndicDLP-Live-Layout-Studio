import os
import shutil
from sklearn.model_selection import train_test_split
from pathlib import Path

BASE_DIR = Path(__file__).parent
IMG_DIR = BASE_DIR / "data"
TRAIN_DIR = IMG_DIR / "train"
TEST_DIR = IMG_DIR / "test"
validExt = ('png', 'jpg', 'jpeg', 'webp')
images = []
for files in os.listdir(IMG_DIR):
    if files.lower().endswith(validExt):
        images.append(files)
train_images, test_images = train_test_split(images, test_size = 0.2, random_state = 50)
for img in train_images:
    shutil.copy(os.path.join(IMG_DIR, img), os.path.join(TRAIN_DIR, img))
for img in test_images:
    shutil.copy(os.path.join(IMG_DIR, img), os.path.join(TEST_DIR, img))

print(f"Split into train : {len(train_images)}, test : {len(test_images)}")