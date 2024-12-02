import torch
import torch.nn.functional as F

def compute_peaks(activations: torch.Tensor, m: int = 2, o: int = 2, w: int = 2, delta: float = 0.1):
    """ Compute the peaks of a given activation time series using Vogl's peak picking algorithm """

    # First pad the data to handle boundary conditions
    padded_activations = F.pad(activations, (m, m), value = 0)

    # Then compute peaks using Vogl's criteria
    is_peak = torch.ones(2 * m + 1, dtype=torch.bool)
    for i in range(1, m + 1):
        is_peak &= activations > padded_activations[i:-i]
        is_peak &= activations > padded_activations[i:-i]
    
    torch.nn.Unfold