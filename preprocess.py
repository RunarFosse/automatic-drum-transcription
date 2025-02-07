import torch
from torch.utils.data import DataLoader

def compute_infrequency_weights(dataloader: DataLoader) -> torch.Tensor:
    """ Compute the infrequency weight of an instrument, given as the 'inverse estimated entropy of their event activity distribution'. """
    n_classes = None
    probabilities = None

    n_timesteps = 0.0
    for _, labels in dataloader:
        # If probability tensor is not defined, count number of classes and do so
        if probabilities == None:
            n_classes = torch.tensor(labels.shape[-1])
            probabilities = torch.zeros(n_classes)
        
        n_timesteps += torch.prod(torch.tensor(labels.shape[:-1]))
        probabilities += torch.sum(labels == 1.0, dim=(0, 1))
    
    # Divide to finish computing probabilities
    probabilities /= n_classes * n_timesteps

    # To prevent Nan's, set probabilities of 0 to a very low number
    probabilities = torch.where(probabilities == 0, torch.tensor(1e-10), probabilities)

    # And compute final weights
    weights = 1.0 / (-probabilities * torch.log(probabilities) - (1 - probabilities) * torch.log(1 - probabilities))
    return weights