import torch
import torch.nn as nn
import torch.nn.functional as F

class FrameSynchronousCNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1)
        self.batchnorm2 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(kernel_size=(1, 3))
        self.dropout1 = nn.Dropout2d(p=0.3)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1)
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


class RNNDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bigrus = nn.ModuleList([nn.GRU(576, 576 // 2, 60, bidirectional=True) for _ in range(3)])
        self.fc = nn.Linear(576, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for bigru in self.bigrus:
            out, _ = bigru(x)

        out = F.sigmoid(self.fc(out))
        return out
    
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
    def __init__(self, n_heads: int):
        super().__init__()
        self.layer_norm = nn.LayerNorm(576)
        self.attentions = nn.ModuleList([nn.MultiheadAttention(embed_dim=576, num_heads=6) for _ in range(n_heads)])
        self.dropout = nn.Dropout(p = 0.1)

        self.fc1 = nn.Linear(576, 4 * 576)
        self.fc2 = nn.Linear(4 * 576, 576)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.layer_norm(x)

        # Transpose before attention
        x = x.transpose(0, 1)

        for attention in self.attentions:
            out, _ = attention(out, out, out)

        # Transpose back
        x = x.transpose(0, 1)

        out = self.dropout(out)

        out = out + x
        out = F.relu(self.fc1(out))
        return self.fc2(out)

class AttentionDecoder(nn.Module):
    def __init__(self, n_heads: int = 5, n_layers: int = 5):
        super().__init__()
        self.positional_encoding = PositionalEncoding(d_model=576)
        self.layers = nn.ModuleList([AttentionLayer(n_heads=n_heads) for _ in range(n_layers)])
        self.fc = nn.Linear(576, 5)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.positional_encoding(x)

        for layer in self.layers:
            out = layer(out)

        return self.fc(out)


class ADTOF_FrameRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = RNNDecoder()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 3, 1), start_dim=2)
        return self.decoder(latent)
    
class ADTOF_FrameAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = FrameSynchronousCNNEncoder()
        self.decoder = AttentionDecoder()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encoder(x)
        latent = torch.flatten(latent.permute(0, 2, 1, 3), start_dim=2)
        return self.decoder(latent)