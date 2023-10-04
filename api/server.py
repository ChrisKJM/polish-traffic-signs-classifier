import os
from PIL import Image
from flask import *
from detect_and_classify import detect_and_classify
from ultralytics import YOLO
import torch

exts = [".jpg", ".jpeg", ".png"]

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the models
detector_model = YOLO("../detector.pt")
classifier_model = torch.load("../classifier.pt").to(device)


@app.route('/detect_and_classify', methods=['GET'])
def detect_and_classify_images():
    if request.method == 'GET':
        try:
            files = request.files.getlist("file")
            if len(files) == 0:
                return {"message": "No file attached to the request"}, 400

            images = dict()

            for file in files:
                if os.path.splitext(file.filename)[-1].lower() not in exts:
                    return {"message": "Allowed file extensions are .jpg, .jpeg and .png"}, 400
                images[file.filename] = Image.open(file)

            result = detect_and_classify(detector_model, classifier_model, images)
            result["message"] = "Success"

            return result, 200

        except Exception as e:
            return {"message": "An internal error has occured"}, 500

    else:
        return {"message": "Invalid request method"}, 405


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)