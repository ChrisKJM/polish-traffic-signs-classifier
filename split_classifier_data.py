import os
import random
import shutil

train_dir = "datasets/classifier/train"
val_dir = "datasets/classifier/val"
test_dir = "datasets/classifier/test"
images_dir = "datasets/classifier/cut_out"

labels = os.listdir(images_dir)


def copy_images(imgs, label, source_dir, target_dir):
    for img in imgs:
        if not os.path.exists(os.path.join(target_dir, label)):
            os.mkdir(os.path.join(target_dir, label))
        shutil.copy(os.path.join(source_dir, label, img), os.path.join(target_dir, label, img))


for label in labels:
    imgs = os.listdir(os.path.join(images_dir, label))
    random.shuffle(imgs)
    
    # change how many images to take depending on whether the class is 'important'
    if label[-4:] == "-imp":
        imgs = imgs[:min(len(imgs) - 1, 143)]
    elif label == "other" or label[:3] == "B33":
        imgs = imgs[:342]
    else:
        imgs = imgs[:min(len(imgs) - 1, 72)]

    # ~70%
    train_imgs = imgs[:7 * len(imgs) // 10]
    
    # ~10%
    val_imgs = imgs[7 * len(imgs) // 10: 8 * len(imgs) // 10]
    
    # ~20%
    test_imgs = imgs[8 * len(imgs) // 10:]

    # copy the images to their corresponding folders
    copy_images(train_imgs, label, images_dir, train_dir)
    copy_images(val_imgs, label, images_dir, val_dir)
    copy_images(test_imgs, label, images_dir, test_dir)
