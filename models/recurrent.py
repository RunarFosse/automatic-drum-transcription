import torch
import torch.nn as nn
import torch.nn.functional as F


class RNNDecoder(nn.Module):
    def __init__(self, num_layers: int = 3, hidden_size: int = 60, use_gru: bool = True):
        super().__init__()
        if use_gru:
            self.rnns = nn.GRU(input_size=576, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        else:
            self.rnns = nn.LSTM(input_size=576, hidden_size=hidden_size, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2 * hidden_size, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnns(x)
        return self.fc(out)