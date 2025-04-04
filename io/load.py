import torch
import torchaudio
import torch.nn.functional as F
from torchaudio import transforms, functional
import partitura

from pathlib import Path
from typing import Dict, Optional

MS_PER_FRAME = 10

def compute_log_filterbank(sr: int = 44100, n_fft: int = 2048, f_min: int = 20, f_max: int = 20000, f_ref: int = 440, bands_per_octave: int = 12, norm: bool = True) -> torch.Tensor:
    """ Compute a logarithmically spaced filterbank. """
    left = torch.floor(torch.log2(torch.tensor(f_min) / f_ref) * bands_per_octave)
    right = torch.ceil(torch.log2(torch.tensor(f_max) / f_ref) * bands_per_octave)

    # Generate logarithmically spaced frequencies
    frequencies = 440 * torch.pow(torch.tensor(2), (torch.arange(left, right) / bands_per_octave))

    # Remove any outside the bounds (due to use of torch.floor/ceil)
    valid_frequencies = torch.logical_and(f_min <= frequencies , frequencies <= f_max)
    frequencies = frequencies[valid_frequencies]

    # Compute Fourier Transform bins
    bins = torch.fft.rfftfreq(n=n_fft, d=1/sr)

    # And compute the indices of these bins, which each filter should cover
    indices = torch.searchsorted(bins, frequencies).clip(1, bins.shape[0] - 1)
    left, right = bins[indices - 1], bins[indices]
    indices -= (frequencies - left < right - frequencies).int()
    indices = torch.unique(indices)

    # Create the filterbank
    filterbank = torch.zeros(size=(indices.shape[0] - 2, bins.shape[0]))

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

def compute_log_filter_spectrogram(waveform: torch.Tensor, sr: int = 44100, n_fft: int = 2048, win_length: int = 2048, f_min: int = 20, f_max: int = 20000, power: int = 1, norm: bool = True) -> torch.Tensor:
    """ Given a waveform, compute a spectrogram with a logarithmic filterbank applied. """

    # If the waveform is stereo, turn it to mono
    if waveform.shape[0] == 2:
        waveform = waveform.mean(dim=0)
 
    # Compute a logarithmically spaced filterbank
    filterbank = compute_log_filterbank(sr=sr, n_fft=n_fft, f_min=f_min, f_max=f_max, norm=norm)
    
    # Then, compute STFT
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=sr // 100, power=power)(waveform)

    # Apply logarithimic filters
    spectrogram = filterbank @ spectrogram

    # Compute log-magnitude
    spectrogram = torch.log10(spectrogram + 1)

    return spectrogram

def readAudio(path: Path, accompaniment: Optional[Path] = None) -> torch.Tensor:
    """ Read an audio file (.wav) into a torch tensor as a mel-spectrogram. """

    # Read the data into mono
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)

    # If accompaniement is set, add to waveform
    if accompaniment:
        accompaniment, _ = torchaudio.load(accompaniment)
        waveform += accompaniment.mean(dim=0)

    # Pad the waveform with zeroes, to be divisible with 4s (400 timeframes) intervals
    samples = torch.tensor(waveform.shape[0])
    timeframes = 1 + torch.floor(samples / (sr // 100))
    padding = (torch.ceil(timeframes / 400) * 400 - 1) * (sr // 100) - samples
    waveform = F.pad(waveform, (0, int(padding)), mode="constant", value=0)

    # Turn it into a logarithmically filtered spectrogram
    spectrogram = compute_log_filter_spectrogram(waveform, sr=sr)

    # Return it on the shape (timesteps, bins), with a last filter dimension
    return spectrogram.T.unsqueeze(-1)
    

def readAnnotations(path: Path, mapping: Dict[str, int], num_frames: int, num_labels: int) -> torch.Tensor:
    """ Read an annotation file into a torch tensor. """

    # Store label and frame indices in lists
    frame_indices = []
    label_indices = []

    # Open the file
    with open(path, "r") as f:
        for line in f.readlines():
            # Parse line
            [time, event] = line.strip().split(" ")

            # If the event ends in a "-" or a digit, remove it
            if event[-1] in ["-"] or event[-1].isnumeric():
                event = event[:-1]

            # Turn into frames and labels
            frame = int(round(float(time) / MS_PER_FRAME * 1000))
            label = mapping[event]
            if label is None:
                # Skip if label is None
                continue

            frame_indices.append(frame)
            label_indices.append(label)
    
    # Turn into torch tensor
    tensor = torch.sparse_coo_tensor(
        torch.tensor([frame_indices, label_indices]),
        torch.ones(len(frame_indices)),
        (num_frames, num_labels)
    ).to_dense()

    # And perform target widening
    adjacents = F.max_pool1d(tensor.T, kernel_size=3, stride=1, padding=1).T - tensor
    tensor += adjacents * 0.5

    return tensor

def readMidi(path: Path, mapping: Dict[str, int], num_frames: int, num_labels: int) -> torch.Tensor:
    """ Read a midi-annotation file into a torch tensor. """

    # Load the midi into list of notes
    notes = partitura.load_performance_midi(path)[0].note_array()

    # Store label and frame indices in lists
    frame_indices = []
    label_indices = []
            
    # Iterate every note
    for note in notes:
        time, pitch = note[0], note[4]

        # Turn into frames and labels
        frame = int(round(float(time) / MS_PER_FRAME * 1000))

        # Extract label from pitch
        label = mapping[pitch]
        if label is None:
            # Skip if label is None
            continue
        
        frame_indices.append(frame)
        label_indices.append(label)

    # Turn into torch tensor
    tensor = torch.sparse_coo_tensor(
        torch.tensor([frame_indices, label_indices]),
        torch.ones(len(frame_indices)),
        (num_frames, num_labels)
    ).to_dense()

    # And perform target widening
    adjacents = F.max_pool1d(tensor.T, kernel_size=3, stride=1, padding=1).T - tensor
    #tensor += adjacents * 0.5

    return tensor