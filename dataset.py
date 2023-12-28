import pandas
from PIL import Image
from torch.utils.data import Dataset


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
