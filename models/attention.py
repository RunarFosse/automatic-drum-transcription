import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    
class AttentionLayer(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=576, num_heads=num_heads, batch_first=True)
        self.dropout = nn.Dropout(p = 0.1)
        self.norm1 = nn.LayerNorm(576)

        self.fc1 = nn.Linear(576, 4 * 576)
        self.fc2 = nn.Linear(4 * 576, 576)
        self.norm2 = nn.LayerNorm(576)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.attention(x, x, x)
        out1 = self.norm1(self.dropout(out1) + x)

        out2 = F.gelu(self.fc1(out1))
        out2 = self.fc2(out2)
        out2 = self.norm2(self.dropout(out2) + out1)
        return out2

class AttentionDecoder(nn.Module):
    def __init__(self, num_heads: int = 6, num_layers: int = 5):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model=576)
        self.layers = nn.Sequential(*[AttentionLayer(num_heads=num_heads) for _ in range(num_layers)])
        self.fc = nn.Linear(576, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.positional_encoding(x)
        out = self.layers(out)

        return self.fc(out)