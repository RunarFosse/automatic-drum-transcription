import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameSynchronousCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(3, 3), padding=1)
        self.batchnorm2 = nn.BatchNorm2d(8)

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout1 = nn.Dropout2d(p=0.3)

        self.conv3 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1)
        self.batchnorm3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.batchnorm4 = nn.BatchNorm2d(32)

        self.pool2 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout2 = nn.Dropout2d(p=0.3)

    def forward(self, x):
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


class RNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bigrus = nn.ModuleList([nn.GRU(288, 32, 60, bidirectional=True) for _ in range(3)])
        self.fc = nn.Linear(64, 5)
    
    def forward(self, x):
        out = torch.flatten(x.permute(0, 2, 3, 1), start_dim=2)

        for bigru in self.bigrus:
            out, _ = bigru(out)

        out = F.sigmoid(self.fc(out))
        return out


class ADTOF_FrameRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = RNNDecoder()
    
    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)