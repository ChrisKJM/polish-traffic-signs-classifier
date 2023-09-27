import os
import random
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2 as transforms
from classifier_model import ResNet34


EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.015
CLASS_COUNT = 22

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


# a dataset class implementation for the dataset
class SignsDataset(Dataset):
    def __init__(self, dataset_dir, transform=None):
        i = 0
        self.labels_dict = dict()

        # load the labels
        for label in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, label)):
                self.labels_dict[i] = label
                i += 1

        self.dataset_dir = dataset_dir
        self.transform = transform

        self.images = list()
        self.labels = list()
        self.exts = [".jpg", ".jpeg", ".png"]

        # load the images
        self.load_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        image = self.images[i]
        label = self.labels[i]

        # apply the transform before returning
        if self.transform:
            image = self.transform(image)

        return image, label

    def load_images(self):
        # iterate over the different labels
        for i, label in self.labels_dict.items():
            label_path = os.path.join(self.dataset_dir, label)

            for img_name in os.listdir(label_path):
                img_path = os.path.join(label_path, img_name)

                # skip if the img is not an actual image for some reson
                if os.path.isdir(img_path) or os.path.splitext(img_name)[-1].lower() not in self.exts:
                    continue

                image = Image.open(img_path)

                self.images.append(image)
                self.labels.append(i)


# a transform that randomly applies the given transforms with a specified probability
class RandomApplyImageAugmentations(torch.nn.Module):
    def __init__(self, *augmentations: tuple, random_order=True):
        super(RandomApplyImageAugmentations, self).__init__()
        self.augmentations = list(augmentations)
        self.random_order = random_order

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        if self.random_order:
            random.shuffle(self.augmentations)

        for aug, prob in self.augmentations:
            r = random.random()
            if r < prob:
                x = aug(x)

        return x


# define the data resizing, normalization and augmentations
train_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    RandomApplyImageAugmentations(
        (transforms.RandomResizedCrop(size=224, scale=(0.8, 1), ratio=(0.8, 1.25)), 0.3),
        (transforms.RandomRotation(degrees=30), 0.25),
        (transforms.ElasticTransform(), 0.15),
        (transforms.Grayscale(num_output_channels=3), 0.1),
        (transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25), 0.3),
        (transforms.RandomEqualize(p=1), 0.15)
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# load the data
print("Loading training dataset...")

train_dataset = SignsDataset(dataset_dir="./datasets/classifier/train", transform=train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading validation dataset...")

val_dataset = SignsDataset(dataset_dir="./datasets/classifier/val", transform=val_transform)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Loading testing dataset...")

test_dataset = SignsDataset(dataset_dir="./datasets/classifier/test", transform=val_transform)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

print("Finished loading the data.")


# define the model
model = ResNet34().to(device)

# model = torch.load("classifier.pt").to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# goes through a single iteration of training and prints metrics based on training data
def train(dataloader, model, loss_func, optimizer):
    size = len(dataloader.dataset)
    batches_count = len(dataloader)
    model.train()
    correct_counts = list()
    total_counts = list()

    for i in range(CLASS_COUNT):
        correct_counts.append(0)
        total_counts.append(0)

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = loss_func(pred, y)

        # sum up the loss values
        loss += loss_func(pred, y).item()

        # count the correct and total counts for each class (to calculate accuracy for every class)
        pred_indexes = pred.argmax(1)
        for i in range(y.shape[0]):
            correct_counts[y[i].item()] += int(y[i] == pred_indexes[i])
            total_counts[y[i].item()] += 1

        # back-propagate
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # calculate the metrics
    correct_count = np.sum(correct_counts)
    avg_loss = loss / batches_count
    accuracy = 100 * correct_count / size
    class_based_accuracies = {dataloader.dataset.labels_dict[i]: correct_counts[i] / total_counts[i] for i in range(CLASS_COUNT)}

    model.training_metrics_history.append((avg_loss, accuracy, class_based_accuracies))

    print("Training:")
    print(f"Average loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f} %")
    print("Class based accuracies: ")
    print(class_based_accuracies)


# displays metrics based on validation data
def test(dataloader, model, loss_func):
    size = len(dataloader.dataset)
    batches_count = len(dataloader)
    model.eval()
    loss = 0
    correct_counts = list()
    total_counts = list()

    for i in range(CLASS_COUNT):
        correct_counts.append(0)
        total_counts.append(0)

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            pred = model(x)

            # sum up the loss values
            loss += loss_func(pred, y).item()

            # count the correct and total counts for each class (to calculate accuracy for every class)
            pred_indexes = pred.argmax(1)
            for i in range(y.shape[0]):
                correct_counts[y[i].item()] += int(y[i] == pred_indexes[i])
                total_counts[y[i].item()] += 1

    # calculate the metrics
    correct_count = np.sum(correct_counts)
    avg_loss = loss / batches_count
    accuracy = 100 * correct_count / size
    class_based_accuracies = {dataloader.dataset.labels_dict[i]: correct_counts[i] / total_counts[i] for i in range(CLASS_COUNT)}

    model.validation_metrics_history.append((avg_loss, accuracy, class_based_accuracies))

    print("Validation:")
    print(f"Average loss: {avg_loss:.3f}, Accuracy: {accuracy:.3f} %")
    print("Class based accuracies: ")
    print(class_based_accuracies)


# main training loop
for e in range(EPOCHS):
    print("")
    print(f"Epoch {e}:")
    train(train_dataloader, model, loss_func, optimizer)
    test(val_dataloader, model, loss_func)

    torch.save(model, f"./classifier_models/model_epoch_{e}.pt")

test(test_dataloader, model, loss_func)
