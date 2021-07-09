# resnet18 for work with the suresort data
# model input: 2x224x224
# model output: 2 classes
import torchvision.models as models
from torch import nn

class ResnetSuresort(nn.Module):
    def __init__(self, n_classes=2, pretrained=True):
        super(ResnetSuresort, self).__init__()
        # Add custom new layer to handle the 2channel input issue
        self.custom_first_layer = nn.Conv2d(in_channels=2,
                                            out_channels=3,
                                            kernel_size=3,
                                            padding=1,
                                            dilation=1)
        # resnet18 as the body of the model
        self.resnet = models.resnet18(pretrained=pretrained)

        # adapt the head/ final layer to our binary case
        self.resnet.fc = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x):
        # solves the 2 channel issue - reshape the input via a conv2d layer
        x = self.custom_first_layer(x)
        x = self.resnet(x)
        return x