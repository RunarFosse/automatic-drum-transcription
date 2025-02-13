import torch
from torch.utils.data import DataLoader
from torchvision import transforms, ops
from pathlib import Path
from typing import Tuple


def compute_normalization(train_path: Path, batch_size: int = 1) -> Tuple[torch.Tensor]:
    """ Compute the normalization terms, (mean, std), of the training dataset. """
    # Insert the training dataset into a dataloader
    train_loader = DataLoader(torch.load(train_path), shuffle=True, batch_size=batch_size, num_workers=16)

    # Compute number of batches
    num_batches = len(train_loader)

    # And compute values
    mean, std = torch.zeros(1), torch.zeors(1)
    for features, _ in train_loader:
        mean = torch.mean(features, dim=(0, 1, 2))
        std = torch.std(features, dim=(0, 1, 2))

    # Return divided over number of batches
    return mean / num_batches, std / num_batches

def create_transform(mean: torch.Tensor, std: torch.Tensor, channels_last: bool) -> transforms.Compose:
    """ Create a preprocessing transforms pipeline. """
    # Normalize the data
    transforms = [
        transforms.Normalize(mean=mean, std=std),
        ]

    # Permute the images if channels_last is set to True
    if channels_last:
        transforms.append(ops.Permute((0, 3, 1, 2)))

    return transforms.Compose(transforms)


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