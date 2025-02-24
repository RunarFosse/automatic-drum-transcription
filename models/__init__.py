""" Expose PyTorch models """

import torch
import torch.nn as nn
import torch.nn.functional as F

from .convolutional import FrameSynchronousCNNEncoder
from .recurrent import RNNDecoder
from .attention import AttentionDecoder


class ADTOF_FrameRNN(nn.Module):
    def __init__(self, num_layers: int = 3, hidden_size: int = 60, use_gru: bool = True):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = RNNDecoder(num_layers=num_layers, hidden_size=hidden_size, use_gru=use_gru)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.decoder(latent)
    
    
class ADTOF_FrameAttention(nn.Module):
    def __init__(self, num_heads: int = 6, num_layers: int = 5):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = AttentionDecoder(num_heads=num_heads, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.decoder(latent)