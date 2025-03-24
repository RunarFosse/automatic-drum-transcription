""" Expose PyTorch models """

import torch
import torch.nn as nn
import torch.nn.functional as F

from ray import tune

from .convolutional import FrameSynchronousCNNEncoder
from .recurrent import RNNDecoder
from .attention import AttentionDecoder, PatchEmbedding

from typing import Tuple


class CNN(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_size: int = 288):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.dense = nn.Sequential(
            nn.Linear(576, hidden_size), 
            nn.ReLU(), 
            *[layer for _ in range(num_layers) for layer in [nn.Linear(hidden_size, hidden_size), nn.ReLU()]]
        )
        self.fc = nn.Linear(hidden_size, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.fc(self.dense(latent))
    
    hyperparameters = {
        "num_layers": tune.grid_search([2, 3, 4]),
        "hidden_size": tune.grid_search([72, 144, 288, 576])
    }

class ADTOF_FrameRNN(nn.Module):
    def __init__(self, num_layers: int = 3, hidden_size: int = 288, use_gru: bool = True):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = RNNDecoder(num_layers=num_layers, hidden_size=hidden_size, use_gru=use_gru)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.decoder(latent)
    
    hyperparameters = {
        "num_layers": tune.grid_search([2, 3, 4, 5]),
        "hidden_size": tune.grid_search([72, 144, 288]),
        "use_gru": tune.grid_search([True, False])
    }
    
    
class ADTOF_FrameAttention(nn.Module):
    def __init__(self, num_heads: int = 6, num_layers: int = 5):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = AttentionDecoder(num_heads=num_heads, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.decoder(latent)

    hyperparameters = {
        "num_heads": tune.grid_search([2, 4, 6, 8]),
        "num_layers": tune.grid_search([2, 4, 6, 8])
    }
    
class VisionTransformer(nn.Module):
    def __init__(self, patch_size: Tuple[int, int] = (1, 21), num_heads: int = 6, num_layers: int = 5):
        super().__init__()
        self.patch_embedding = PatchEmbedding(patch_size=patch_size)
        self.decoder = AttentionDecoder(num_heads=num_heads, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.patch_embedding(x)
        return self.decoder(latent)
    
    hyperparameters = {
        "patch_size": tune.grid_search([(1, 7), (1, 14), (1, 21)]),
        "num_heads": tune.grid_search([4, 6, 8]),
        "num_layers": tune.grid_search([6, 8, 10])
    }