import sys
import os
import streamlit as st

# so that detect_and_classify.py is visible to the program
@st.cache_resource
def append_project_folder():
    sys.path.append(os.path.abspath("."))
    return None

append_project_folder()
from detect_and_classify import detect_and_classify, CLASSES_FULL
from PIL import Image, ImageDraw, ImageOps


img = st.file_uploader(label="Select an image", type=["png", "jpg"])

if img is not None:
    image = Image.open(img).convert("RGB")
    image = ImageOps.exif_transpose(image)

    result = detect_and_classify({"image": image})["image"]

    scale = max(image.width, image.height) / 256
    image = image.resize((int(image.width / scale), int(image.height // scale)))

    # sort by y, then by x
    result.sort(key=lambda val: (val["box"][1], val["box"][0]))
    draw = ImageDraw.Draw(image)

    text = ""
    i = 1
    for sign in result:
        draw.rectangle(tuple([sign["box"][i]/scale for i in range(4)]), width=5, outline=(255, 0, 0))
        text += f"{i}. {CLASSES_FULL[sign['prediction']]}\n"
        i += 1

    if len(result) == 0:
        text = "No signs detected"

    st.image(image)
    st.text(text)
    image.close()
