import os
import random
import shutil

images_dir = "datasets/detector/imgs/"
labels_dir = "datasets/detector/labels/"

train_dir = "datasets/detector/train/"
val_dir = "datasets/detector/val/"
test_dir = "datasets/detector/test/"

data = os.listdir(images_dir)
random.shuffle(data)

# 70-10-20 split
train_data = data[:7 * len(data) // 10]
val_data = data[7 * len(data) // 10: 8 * len(data) // 10]
test_data = data[8 * len(data) // 10:]


def copy_data(data, target_dir):
    # assuming the label file name is the same as image file name
    for image_file_name in data:
        image_path = os.path.join(images_dir, image_file_name)
        label_path = os.path.join(labels_dir, os.path.splitext(image_file_name)[0] + ".txt")
        shutil.copy(image_path, os.path.join(target_dir))
        shutil.copy(label_path, target_dir)


copy_data(train_data, train_dir)
copy_data(val_data, val_dir)
copy_data(test_data, test_dir)
