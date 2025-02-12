import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameSynchronousCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout1 = nn.Dropout2d(p=0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False)
        self.batchnorm4 = nn.BatchNorm2d(64)

        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout2 = nn.Dropout2d(p=0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = self.batchnorm1(out)
        out = F.relu(self.conv2(out))
        out = self.batchnorm2(out)
        out = self.dropout1(self.pool1(out))

        out = F.relu(self.conv3(out))
        out = self.batchnorm3(out)
        out = F.relu(self.conv4(out))
        out = self.batchnorm4(out)
        out = self.dropout2(self.pool2(out))

        return out