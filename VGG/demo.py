import vgg_model
import torch

model = vgg_model.VGG_Net(input_channels=3, number_of_classes=1000)
X = torch.randn(1, 3, 224, 224)
print(model(X).shape)
