import torch.nn as nn
from torchvision import models

class Resnet_Encoder(nn.Module):
    def __init__(self, criterion):
        super().__init__()
        self.criterion = criterion
        model = models.resnet50(pretrained=True)
        #remove last layer to get encoder
        modules = list(model.children())[:-1]
        self.resnet_model = nn.Sequential(*modules)
    def forward(self, x):
        return self.resnet_model(x)