import torch
import json
import pandas
from torch import nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


IMAGES_DIR = "data\\images\\"
TRAIN_DATA_PATH = "data\\train.csv"
TEST_DATA_PATH = "data\\test.csv"

with open("data\\classdict.json", "r") as c, open("data\\labeldict.json", "r") as l:
    CLASS_DICT = json.load(c)
    LABEL_DICT = json.load(l)

IMAGE_SIZE = 224


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

    def __call__(self, path):
        image = Image.open(path)
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


training_data = FoodDataset(TRAIN_DATA_PATH, IMAGES_DIR, FormatImage())
test_data = FoodDataset(TEST_DATA_PATH, IMAGES_DIR, FormatImage())


train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
