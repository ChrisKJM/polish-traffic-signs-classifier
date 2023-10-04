from torch import nn
from ultralytics import YOLO
from torchvision.transforms import v2 as transforms
import torch
from classifier_model import ResNet34, ResidualBlock


device = "cuda" if torch.cuda.is_available() else "cpu"


softmax = nn.Softmax()

# load the models
detector_model = YOLO("detector.pt")
classifier_model = torch.load("classifier.pt").to(device)

# resize and normalize transform
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


def detect(images: dict()) -> dict:
    # detect the images
    detection_results = detector_model.predict(list(images.values()), imgsz=640, conf=0.5)

    results = dict()

    # iterate over the detection results
    for image_name, detection_result in zip(images.keys(), detection_results):
        results[image_name] = list()

        # append to results
        for box in detection_result.boxes:
            results[image_name].append({"box": box.xyxy[0].tolist(), "box_confidence": box.conf.item()})

    return results


def classify(images: dict()) -> dict:
    results = dict()

    # iterate over the image names
    for image_name, cut_out_images in images.items():
        results[image_name] = list()
        inputs = list()

        # if no images, go to the next one
        if len(cut_out_images) == 0:
            continue

        # preprocess the image
        for cut_out_image in cut_out_images:
            inputs.append(transform(cut_out_image))

        # convert the list of images into tensor
        inputs = torch.stack(inputs).to(device)

        # classify
        output = classifier_model(inputs)
        predictions = output.argmax(axis=1)

        # calculate confidences
        pred_confidences = softmax(output)[torch.arange(output.size()[0]), predictions]

        # append to results
        for pred, pred_confidence in zip(predictions, pred_confidences):
            results[image_name].append({"prediction": pred.item(), "prediction_confidence": pred_confidence.item()})

    return results


def detect_and_classify(images: dict()) -> dict:
    results = dict()

    # detect
    detection_results = detect(images)

    cut_out_images = dict()

    # for each image, cut out the image
    for image_name, boxes in detection_results.items():
        cut_out_images[image_name] = list()

        for box in boxes:
            cut_out_images[image_name].append(
                images[image_name].crop(box["box"])
            )

    # classify
    classification_results = classify(cut_out_images)

    # combine the results
    for image_name in images.keys():
        results[image_name] = list()
        for box, cls in zip(detection_results[image_name], classification_results[image_name]):
            results[image_name].append(
                {
                    "box": box["box"], "box_confidence": box["box_confidence"],
                    "prediction": cls["prediction"], "prediction_confidence": cls["prediction_confidence"]
                 }
            )

    return results


