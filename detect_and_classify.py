import os
from ultralytics import YOLO
from PIL import Image
from torchvision.transforms import v2 as transforms
import torch
from classifier_model import ResNet34, ResidualBlock


source_dir = "images"

device = "cuda" if torch.cuda.is_available() else "cpu"


# define the classes
# classes = ["A1", "A2", "A7", "A17-imp", "A21", "A30", "B1", "B2", "B20-imp", "B21", "B22", "B23", "B33-30-50",
#            "B33-60+", "B36", "B41", "C2", "C4", "C12", "D1-imp", "D6-imp", "other"]

classes = ["niebezpieczny zakręt w prawo", "niebezpieczny zakręt w lewo", "ustąp pierwszeństwa", "dzieci (!)",
           "tramwaj", "inne niebezpieczeństwo", "zakaz ruchu w obu kierunkach", "zakaz wjazdu", "stop (!)",
           "zakaz skręcania w lewo", "zakaz skręcania w prawo", "zakaz zawracania", "ograniczenie prędkości 30 - 50",
           "ograniczenie prędkości 60+", "zakaz zatrzymywania się", "zakaz ruchu pieszych", "nakaz jazdy w prawo za znakiem",
           "nakaz jazdy w lewo za znakiem", "ruch okrężny", "droga z pierwszeństwem (!)", "przejście dla pieszych (!)",
           "inne"]

exts = [".jpg", ".jpeg", ".png"]

# load the models
detector_model = YOLO("detector.pt")
classifier_model = torch.load("classifier.pt").to(device)

# get the images
images = os.listdir(source_dir)
images = [os.path.join(source_dir, images[i]) for i in range(len(images)) if os.path.splitext(images[i])[-1].lower() in exts]

# detect the images
results = detector_model.predict(images, imgsz=640, conf=0.5)


# resize and normalize transform
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

for result in results:
    segmented_images = list()

    # turn into pillow image (while changing the BGR array into RGB)
    img = Image.fromarray(result.orig_img[:, :, ::-1], mode="RGB")

    # cut out the detected boxes
    for box in result.boxes:
        coords = box.xyxy[0].tolist()
        seg = img.crop(coords)
        segmented_images.append(transform(seg))

    # print(len(segmented_images))

    # convert the list of images into tensor
    segmented_images = torch.stack(segmented_images).to(device)

    # classify
    output = classifier_model(segmented_images)
    predictions = output.argmax(axis=1)

    # show detected signs and print their classes
    Image.fromarray(result.plot()[:,:,::-1]).show()
    print([classes[pred] for pred in predictions])
    input("Press enter for next image...")
