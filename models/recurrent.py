import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    def __init__(self, num_layers: int = 3):
        super().__init__()
        self.bigrus = nn.GRU(576, 60, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * 60, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.bigrus(x)
        return self.fc(out)