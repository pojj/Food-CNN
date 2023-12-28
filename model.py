import torch
from torch import nn
import torchvision.models as models


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # resnet = models.resnet101(weights="IMAGENET1K_V2")
        resnet = models.resnet50(weights="IMAGENET1K_V2")
        # resnet = models.resnet34(weights="IMAGENET1K_V1")
        # resnet = models.resnet18(weights="IMAGENET1K_V1")

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

        self.fc = torch.nn.Linear(2048, 101)
        # self.fc = torch.nn.Linear(512, 101)

    def forward(self, x):
        x = self.residual(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
