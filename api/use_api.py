import requests
import os

source_dir = "../images"
exts = [".jpg", ".jpeg", ".png"]

image_paths = os.listdir(source_dir)
image_paths = [os.path.join(source_dir, image_path) for image_path in image_paths if os.path.splitext(image_path)[-1].lower() in exts]

images = [("file", open(image_path, "rb")) for image_path in image_paths]

response = requests.get("http://localhost:5000/detect_and_classify", files=images)
print(response.text)
