import torch
from torch.utils.data import DataLoader

def compute_infrequency_loss(dataloader: DataLoader) -> torch.Tensor:
    n_classes = None
    probabilities = None

    n_timesteps = 0.0
    for _, labels in dataloader:
        # If probability tensor is not defined, count number of classes and do so
        if not probabilities:
            n_classes = labels.shape[-1]
            probabilities = torch.zeros(n_classes)
        
        n_timesteps += torch.sum(labels.shape[:-1])
        probabilities += torch.sum(labels, dim=(0, 1))
    
    # Divide to finish computing probabilities
    probabilities /= n_classes * n_timesteps

    # And compute final weights
    weights = 1.0 / (-probabilities * torch.log(probabilities) - (1 - probabilities) * torch.log(1 - probabilities))
    return weights