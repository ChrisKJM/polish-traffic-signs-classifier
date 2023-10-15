import sys
import os
import streamlit as st

# so that detect_and_classify.py is visible to the program
@st.cache_resource
def append_project_folder():
    sys.path.append(os.path.abspath("."))
    print(sys.path)
    return None

append_project_folder()
from detect_and_classify import detect_and_classify, CLASSES_FULL
from PIL import Image, ImageDraw
import json


img = st.file_uploader(label="Select an image", type=["png", "jpg"])

if img is not None:
    image = Image.open(img)
    result = detect_and_classify({"image": image})["image"]
    draw = ImageDraw.Draw(image)
    text = ""
    i = 1
    for sign in result:
        draw.rectangle(sign["box"], width=7, outline=(255, 0, 0))
        text += f"{i}. {CLASSES_FULL[sign['prediction']]}\n"
        i += 1
    st.image(image)
    st.text(text)
    image.close()
