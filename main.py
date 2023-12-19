import torch
import json
import pandas
import time
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torchvision.models as models


IMAGES_DIR = "data\\images\\"
TRAIN_DATA_PATH = "data\\train.csv"
TEST_DATA_PATH = "data\\test.csv"

with open("data\\classdict.json", "r") as c, open("data\\labeldict.json", "r") as l:
    CLASS_DICT = json.load(c)
    LABEL_DICT = json.load(l)

IMAGE_SIZE = 32

LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 10


class FoodDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform, target_transform=None):
        self.img_labels = pandas.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = Image.open(self.img_dir + self.img_labels.iloc[idx, 1])
        label = self.img_labels.iloc[idx, 0]

        image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label


class FormatImage:
    def __init__(self):
        self.to_tensor = ToTensor()

    def __call__(self, image):
        width, height = image.size
        max_dimension = max(width, height)

        scale_factor = IMAGE_SIZE / max_dimension

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        scaled_image = image.resize((new_width, new_height))

        new_image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        padding = ((IMAGE_SIZE - new_width) // 2, (IMAGE_SIZE - new_height) // 2)
        new_image.paste(scaled_image, padding)

        return self.to_tensor(new_image)


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        resnet = models.resnet101(weights="IMAGENET1K_V2")

        self.residual = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )

        self.fc = nn.Linear(2048, 101)

    def forward(self, x):
        x = self.residual(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train_loop(dataloader, model, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    t0 = time.time()
    first = True

    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Print time calcuations for only first batch
        if first:
            projected = int((time.time() - t0) * len(dataloader))
            days = projected // 86400
            hours = (projected - (days * 86400)) // 3600
            minutes = (projected - (days * 86400) - (hours * 3600)) // 60
            print(f"Batch time: {time.time()-t0:>0.1f}s,", end=" ")
            print(
                f"Projected epoch time: {days} days, {hours} hours, {minutes} minutes"
            )
            first = False

        # Print info every couple batches
        if (batch + 1) % 74 == 0:
            loss = loss.item()
            if (batch + 1) < len(dataloader):
                current = (batch + 1) * BATCH_SIZE
            else:
                current = size
            print(f"Loss: {loss:>7f}, [{current:>5d}/{size:>5d}],", end=" ")
            print(f"Elapsed time: {time.time()-t0:>0.1f}s")


def test_loop(dataloader, model, loss_fn):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    t0 = time.time()

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).int().sum().item()

    test_loss /= num_batches
    correct /= size

    print("Test Error:")
    print(f"Accuracy: {(100*correct):>0.1f}%,", end=" ")
    print(f"Avg loss: {test_loss:>8f},", end=" ")
    print(f"Test time: {time.time()-t0:>0.1f}s\n")


model = NeuralNetwork()

for param in model.residual.parameters():
    param.requires_grad_(False)


training_data = FoodDataset(TRAIN_DATA_PATH, IMAGES_DIR, FormatImage())
test_data = FoodDataset(TEST_DATA_PATH, IMAGES_DIR, FormatImage())

train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    torch.save(model.state_dict(), f"foodweights{t+1}.pth")
    test_loop(test_dataloader, model, loss_fn)

print("Done!")
