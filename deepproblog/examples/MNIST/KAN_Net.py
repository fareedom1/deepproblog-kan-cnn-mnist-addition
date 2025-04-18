# File KAN_Net.py
import torch
import torch.nn as nn
from .conv_kan_baseline import SimpleConvKAN  # Adjust the import if your file structure differs

class KAN_Net(nn.Module):
    def __init__(self):
        super(KAN_Net, self).__init__()
        # Define layer sizes for the KAN network.
        # These numbers are illustrative; you might need to tweak them based on your experiments.
        layer_sizes = [32, 64, 128, 256]
        
        # Instantiate the SimpleConvKAN model with MNIST-compatible parameters.
        self.model = SimpleConvKAN(
            layer_sizes=layer_sizes,
            num_classes=10,       # MNIST has 10 classes (digits 0-9)
            input_channels=1,     # MNIST images are grayscale (1 channel)
            spline_order=3,
            degree_out=2,         # Using a linear output (set to  2 to use a linear layer)
            groups=1,
            dropout=0.0,
            dropout_linear=0.0,
            l1_penalty=0.0,
            affine=True,
            norm_layer=nn.BatchNorm2d
        )

    def forward(self, x):
        return self.model(x)
