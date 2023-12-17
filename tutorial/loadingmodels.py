import torch
from torch import nn
import torchvision.models as models

# model = models.vgg16(weights="IMAGENET1K_V1")
# torch.save(model.state_dict(), "tutorial\models\model_weights.pth")

model = models.vgg16()  # we do not specify ``weights``, i.e. create untrained model
model.load_state_dict(torch.load("tutorial\models\model_weights.pth"))

print(model)
