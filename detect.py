import os
from ultralytics import YOLO
from PIL import Image


source_dir = "images"
save_dir = "images_cut_out"

exts = [".jpg", ".jpeg", ".png"]

# load the model
detector_model = YOLO("detector.pt")

# get the images
images = os.listdir(source_dir)
images = [os.path.join(source_dir, images[i]) for i in range(len(images)) if os.path.splitext(images[i])[-1].lower() in exts]

# detect the images
results = detector_model.predict(images, imgsz=640, conf=0.5)

i = 0
for result in results:
    # turn into pillow image (while changing the BGR array into RGB)
    img = Image.fromarray(result.orig_img[:, :, ::-1], mode="RGB")

    # cut out the detected boxes
    for box in result.boxes:
        coords = box.xyxy[0].tolist()
        seg = img.crop(coords)
        seg.save(os.path.join(save_dir, f"{i}.jpg"))
        i += 1
