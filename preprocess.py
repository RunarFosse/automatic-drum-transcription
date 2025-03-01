import torch
import torchaudio
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
    mean, std = torch.zeros(1), torch.zeros(1)
    for features, _ in train_loader:
        mean += torch.mean(features, dim=(0, 1, 2))
        std += torch.std(features, dim=(0, 1, 2))

    # Return divided over number of batches
    return mean / num_batches, std / num_batches

def create_transform(mean: torch.Tensor, std: torch.Tensor, channels_last: bool) -> transforms.Compose:
    """ Create a preprocessing transforms pipeline. """
    # Normalize the data
    composition = [
        transforms.Normalize(mean=mean, std=std),
        ]

    # Permute the images if channels_last is set to True
    if channels_last:
        composition.append(ops.Permute((0, 3, 1, 2)))

    return transforms.Compose(composition)


def compute_infrequency_weights(dataloader: DataLoader) -> torch.Tensor:
    """ Compute the infrequency weight of an instrument, given as the 'inverse estimated entropy of their event activity distribution'. """
    num_classes = None
    probabilities = None

    num_timesteps = 0.0
    for _, labels in dataloader:
        # If probability tensor is not defined, count number of classes and do so
        if probabilities == None:
            num_classes = torch.tensor(labels.shape[-1])
            probabilities = torch.zeros(num_classes)
        
        num_timesteps += labels.shape[0] * labels.shape[1]
        probabilities += torch.sum(labels == 1.0, dim=(0, 1))
    
    # Divide to finish computing probabilities
    probabilities /= num_timesteps

    # To prevent Nan's, set probabilities of 0 to a very low number
    probabilities = torch.where(probabilities == 0, torch.tensor(1e-10), probabilities)

    # And compute final weights
    weights = 1.0 / (-probabilities * torch.log(probabilities) - (1.0 - probabilities) * torch.log(1.0 - probabilities))
    return weights



def compute_log_filterbank(sr: int = 44100, n_fft: int = 2048, f_min: int = 20, f_max: int = 20000, f_ref: int = 440, bands_per_octave: int = 12, norm: bool = True) -> torch.Tensor:
    left = torch.floor(torch.log2(torch.tensor(f_min) / f_ref) * bands_per_octave)
    right = torch.ceil(torch.log2(torch.tensor(f_max) / f_ref) * bands_per_octave)

    # Generate logarithmically spaced frequencies
    frequencies = 440 * torch.pow(torch.tensor(2), (torch.arange(left, right) / bands_per_octave))

    # Remove any outside the bounds (due to use of torch.floor/ceil)
    valid_frequencies = torch.logical_and(f_min <= frequencies , frequencies <= f_max)
    frequencies = frequencies[valid_frequencies]

    # Compute Fourier Transform bins
    bins = torch.fft.rfftfreq(n=n_fft, d=1/sr)
    indices = torch.unique(torch.searchsorted(bins, frequencies).clip(1, bins.shape[0] - 1))

    # Create the filterbank
    filterbank = torch.zeros(size=(indices.shape[0] - 1, bins.shape[0]))

    # And create each filter
    for i in range(indices.shape[0] - 2):
        # Compute a start and stop index (forced to be bigger than 1)
        start, center, stop = indices[i:i+3]
        if stop - start < 2:
            center = start
            stop = start + 1

        # Set the values
        filterbank[i, start:center] = torch.linspace(0, 1, center  - start)
        filterbank[i, center:stop] = torch.linspace(1, 0, stop - center)

        # Normalize
        if norm:
            filterbank[i, start:stop] /= torch.sum(filterbank[i, start:stop])
    
    return filterbank


def compute_log_filter_spectrogram(waveform: torch.Tensor, sr: int = 44100, n_fft: int = 2048, win_length: int = 2048, hop_length: int = 441, f_min: int = 20, f_max: int = 20000, power: int = 1, norm: bool = True) -> torch.Tensor:
    """ Given a waveform, compute a spectrogram with a logarithmic filterbank applied. """

    # If the waveform is stereo, turn it to mono
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)
 
    # Compute a logarithmically spaced filterbank
    filterbank = compute_log_filterbank(sr=sr, n_fft=n_fft, f_min=f_min, f_max=f_max, norm=norm)
    
    # Then, compute STFT
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length, power=power)(waveform)

    # Apply logarithimic filters
    spectrogram = filterbank @ spectrogram

    # Compute log-magnitude
    spectrogram = torch.log10(spectrogram + 1)

    return spectrogram

def invert_log_filter_spectrogram(spectrogram: torch.Tensor, sr: int = 44100, n_fft: int = 2048, win_length: int = 2048, hop_length: int = 441, f_min: int = 20, f_max: int = 20000, power: int = 1, norm: bool = True) -> torch.Tensor:
    """ Given a log logarithmically filterbank spectrogram, invert it, and return its waveform. """

    # Invert log magnitude
    spectrogram = torch.pow(torch.tensor(10), spectrogram) - 1
    
    # Remove filters from spectrogram
    filterbank = compute_log_filterbank(sr=sr, n_fft=n_fft, f_min=f_min, f_max=f_max)
    inverse_filterbank = torch.linalg.pinv(filterbank)
    spectrogram = inverse_filterbank @ spectrogram

    # Apply Griffin-Lim transform
    griffin_lim_transform = torchaudio.transforms.GriffinLim(power=power, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    waveform = griffin_lim_transform(spectrogram)

    return waveform


def invert_mel_spectrogram(spectrogram: torch.Tensor, n_fft: int = 2048, win_length: int = 2048, hop_length: int = 441, n_mels: int = 84, f_min: int = 20, f_max: int = 20000, norm: str = "slaney", mel_scale: str = "htk", n_iter: int = 32, power: int = 2) -> torch.Tensor:
    """ Given a log mel spectrogram, invert it, and return its waveform. """

    # Sampling rate corresponds to 44.1 khz
    sr = 44100
    n_stft = int((n_fft//2) + 1)

    inverse_transform = torchaudio.transforms.InverseMelScale(sample_rate=sr, n_stft=n_stft, n_mels=n_mels, f_min=f_min, f_max=f_max, norm=norm, mel_scale=mel_scale)
    grifflim_transform = torchaudio.transforms.GriffinLim(power=power, n_iter=n_iter, n_fft=n_fft, win_length=win_length, hop_length=hop_length)

    inverse_waveform = inverse_transform(spectrogram)
    pseudo_waveform = grifflim_transform(inverse_waveform)

    return pseudo_waveform