import os
from PIL import Image
from torchvision.transforms import v2 as transforms
import torch
from classifier_model import ResNet34, ResidualBlock


source_dir = "images_cut_out"

device = "cuda" if torch.cuda.is_available() else "cpu"


# define the classes
# classes = ["A1", "A17-imp", "A2", "A21", "A30", "A7", "B1", "B2", "B20-imp", "B21", "B22", "B23", "B33-30-50",
#            "B33-60+", "B36", "B41", "C12", "C2", "C4", "D1-imp", "D6-imp", "other"]

classes = ["niebezpieczny zakręt w prawo", "dzieci (!)", "niebezpieczny zakręt w lewo", "tramwaj",
           "inne niebezpieczeństwo", "ustąp pierwszeństwa", "zakaz ruchu w obu kierunkach", "zakaz wjazdu", "stop (!)",
           "zakaz skręcania w lewo", "zakaz skręcania w prawo", "zakaz zawracania", "ograniczenie prędkości 30 - 50",
           "ograniczenie prędkości 60+", "zakaz zatrzymywania się", "zakaz ruchu pieszych", "ruch okrężny",
           "nakaz jazdy w prawo za znakiem", "nakaz jazdy w lewo za znakiem", "droga z pierwszeństwem (!)",
           "przejście dla pieszych (!)", "inne"]

exts = [".jpg", ".jpeg", ".png"]

# load the model
classifier_model = torch.load("classifier.pt").to(device)

# get the images
images = os.listdir(source_dir)
images = [os.path.join(source_dir, images[i]) for i in range(len(images)) if os.path.splitext(images[i])[-1].lower() in exts]


# resize and normalize transform
transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

for image_path in images:
    image = Image.open(image_path)
    image.show()
    image = transform(image)

    image = torch.stack((image,)).to(device)

    # classify
    output = classifier_model(image)
    print(output)
    predictions = output.argmax(axis=1)

    print(image_path)
    print([classes[pred] for pred in predictions])
    input("Press enter for next image...")
