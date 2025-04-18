import torch.nn as nn
from .kan_conv import KANConv2DLayer


class MNIST_CNN(nn.Module):
    def __init__(self):
        super(MNIST_CNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )

    def forward(self, x):
        x = x.unsqueeze(0)
        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)
        return x


class MNIST_Classifier(nn.Module):
    def __init__(self, n=10, with_softmax=True):
        super(MNIST_Classifier, self).__init__()
        self.with_softmax = with_softmax
        if with_softmax:
            self.softmax = nn.Softmax(1)
        self.classifier = nn.Sequential(
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x.squeeze(0)


class MNIST_Net(nn.Module):
    def __init__(self, n=10, with_softmax=True, size=16 * 4 * 4):
        super(MNIST_Net, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        if with_softmax:
            if n == 1:
                self.softmax = nn.Sigmoid()
            else:
                self.softmax = nn.Softmax(1)
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.MaxPool2d(2, 2),  # 6 24 24 -> 6 12 12
            nn.ReLU(True),
            nn.Conv2d(6, 16, 5),  # 6 12 12 -> 16 8 8
            nn.MaxPool2d(2, 2),  # 16 8 8 -> 16 4 4
            nn.ReLU(True),
        )
        self.classifier = nn.Sequential(
            nn.Linear(size, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, n),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.size)
        x = self.classifier(x)
        if self.with_softmax:
            x = self.softmax(x)
        return x

class MNIST_KAN(nn.Module):
    def __init__(self):
        super(MNIST_KAN, self).__init__()
        self.encoder = nn.Sequential(
        KANConv2DLayer(in_channels=1, out_channels=6, kernel_size=5, input_dim=1, output_dim=6),
        nn.MaxPool2d(2, 2),
        nn.ReLU(True),
        KANConv2DLayer(in_channels=6, out_channels=16, kernel_size=5, input_dim=6, output_dim=16),
        nn.MaxPool2d(2, 2),
        nn.ReLU(True),
)


    def forward(self, x):
        #x = x.unsqueeze(0)  # just like MNIST_CNN
        x = x if x.dim() == 4 else x.unsqueeze(0)

        x = self.encoder(x)
        x = x.view(-1, 16 * 4 * 4)  # flatten like the CNN
        return x

class MNIST_KAN_Net(nn.Module):
    def __init__(self, n=10, with_softmax=True, size=16 * 4 * 4):
        super(MNIST_KAN_Net, self).__init__()
        self.with_softmax = with_softmax
        self.size = size
        self.encoder = MNIST_KAN()
        self.classifier = MNIST_Classifier(n=n, with_softmax=with_softmax)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
