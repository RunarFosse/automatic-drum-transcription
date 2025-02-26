import torch
import torchaudio
import torch.nn.functional as F
from torchaudio import transforms, functional


from pathlib import Path
from typing import Dict, Optional

MS_PER_FRAME = 10

def readAudio(path: Path, accompaniment: Optional[Path] = None) -> torch.Tensor:
    """ Read an audio file (.wav) into a torch tensor as a mel-spectrogram. """

    # Read the data into mono
    waveform, sr = torchaudio.load(path)
    waveform = waveform.mean(dim=0)


    # If accompaniement is set, add to waveform
    if accompaniment:
        accompaniment, _ = torchaudio.load(accompaniment)
        waveform += accompaniment.mean(dim=0)

    # Pad the waveform with zeroes, to be divisible with 4s intervals
    timesteps = torch.tensor(waveform.shape[0])
    padding = torch.ceil(timesteps / (4.0 * sr)) * (4.0 * sr) - timesteps - 1
    waveform = F.pad(waveform, (0, int(padding)), mode="constant", value=0)

    # Turn it into a mel spectrogram, on the shape
    spectrogram = transforms.MelSpectrogram(sample_rate=sr, n_fft=2048, win_length=2048, hop_length=441, n_mels=84)(waveform)

    # Turn it to log scale
    spectrogram = functional.amplitude_to_DB(spectrogram, multiplier=10, amin=1e-10, db_multiplier=1)

    # Return it on the shape (timesteps, bins)
    return spectrogram.T
    

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
    return tensor