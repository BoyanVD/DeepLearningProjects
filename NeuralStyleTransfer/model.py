import torch
import torch.nn as nn
import torchvision.models as models
import constants

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()

        self.model_features = constants.CHOSEN_MODEL_FEATURES
        self.model = models.vgg19(pretrained=True).features[:int(self.model_features[-1])] # dont need the last layers

    def forward(self, X):
        features = []

        for layer_number, layer in enumerate(self.model):
            X = layer(X)
            if str(layer_number) in self.model_features:
                features.append(X)

        return features
