import torch.nn as nn
from torch_conv_kan import KANConv2d, KANLinear  # assuming you installed and imported correctly
import torch.nn.functional as F

class SimpleConvKAN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleConvKAN, self).__init__()
        self.encoder = nn.Sequential(
            KANConv2d(1, 6, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            KANConv2d(6, 16, kernel_size=5),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            KANLinear(16 * 4 * 4, 120),
            nn.ReLU(),
            KANLinear(120, 84),
            nn.ReLU(),
            KANLinear(84, num_classes),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        x = self.classifier(x)
        return x
