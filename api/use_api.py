import json
import requests
import os
from detect_and_classify import CLASSES_FULL, CLASSES
import numpy as np

source_dir = "../images"
exts = [".jpg", ".jpeg", ".png"]

image_names = os.listdir(source_dir)

images = [("file", open(os.path.join(source_dir, image_name), "rb")) for image_name in image_names if os.path.splitext(image_name)[-1] in exts]

response = requests.get("http://localhost:5000/detect_and_classify", files=images)

results: dict = json.loads(response.text)["results"]


# label files should contain traffic sign codes (like in CLASSES in detect_and_classify.py), one per line
# if class_counts has a negative value, then the model missed out on a sign
# if class_counts has a positive value, then the model predicted a sign that wasn't there
for image_name, result in results.items():
    print(f"Looking at {image_name}...")
    label_name = os.path.splitext(image_name)[0] + ".txt"

    class_counts = np.zeros(len(CLASSES_FULL))

    if os.path.exists(os.path.join(source_dir, label_name)):
        with open(os.path.join(source_dir, label_name), "r") as label_file:
            for label in label_file.readlines():
                class_counts[CLASSES.index(label.strip())] -= 1

    for sign in result:
        class_counts[sign["prediction"]] += 1

    if len(class_counts.nonzero()) == 0:
        print("No mistakes found!")

    for i in class_counts.nonzero()[0].tolist():
        if class_counts[i] > 0:
            print(f"Found {abs(class_counts[i])} instance(s) of {CLASSES_FULL[i]} that weren't on the image")
        elif class_counts[i] < 0:
            print(f"Found {abs(class_counts[i])} instance(s) of {CLASSES_FULL[i]} not detected by the model")
    print()
