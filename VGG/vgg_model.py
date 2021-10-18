import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import constants

class VGG_Net(nn.Module):
    def __init__(self, input_channels=3, number_of_classes=100, architecture=constants.VGG16_ARCHITECTURE):
        super(VGG_Net, self).__init__()
        self.input_channels = input_channels
        self.conv_layers = self.create_conv_layers(architecture)
        self.fully_connected = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, number_of_classes)
        )

    def create_conv_layers(self, architecture):
        layers = []
        input_channels = self.input_channels

        for layer in architecture:
            if type(layer) == int:
                output_channels = layer
                layers += [
                    nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                    nn.BatchNorm2d(layer),
                    nn.ReLU()
                ]
                input_channels = layer
            elif layer == 'M':
                layers += [nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))]

        return nn.Sequential(*layers)

    def forward(self, X):
        X = self.conv_layers(X)
        X = X.reshape(X.shape[0], -1)
        X = self.fully_connected(X)

        return X
