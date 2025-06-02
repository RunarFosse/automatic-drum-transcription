import torch
from torch.utils.data import TensorDataset, DataLoader
from load import readAudio, readMidi
from pathlib import Path
import sys
sys.path.append("../master-thesis")
from preprocess import create_transform
from evaluate import evaluate_model

model_dir = Path(__file__).resolve().parent.parent / "study" / "Dataset" / "Convolutional RNN" / "ENST+MDB+EGMD+SLAKH+ADTOF"
config = torch.load(model_dir / "config.pt")
state_dict = torch.load(model_dir / "model.pt")
model = config["Model"](**config["parameters"])
model.load_state_dict(state_dict)

# Create the transforms
transforms = create_transform(mean=config["transforms"]["mean"], std=config["transforms"]["std"], channels_last=True)

""" Run this file to turn SADTP into a stored PyTorch dataset """

# 5-drum mapping for MIDI, from "https://github.com/khiner/DrumClassification/blob/main/create_label_mapping.py" and "https://soundprogramming.net/file-formats/general-midi-drum-note-numbers/"
MIDI_MAPPING = {
    35: 0, #'Acoustic Bass Drum',
    36: 0, #'Bass Drum',

    37: 1, #'Side Stick',
    38: 1, #'Acoustic Snare',
    39: 1, #'Hand Clap',
    40: 1, #'Electric Snare',

    41: 2, #'Low Floor Tom',
    43: 2, #'High Floor Tom',
    45: 2, #'Low Tom',
    47: 2, #'Low-Mid Tom',
    48: 2, #'Hi-Mid Tom',
    50: 2, #'High Tom',

    42: 3, #'Closed Hi Hat',
    44: 3, #'Pedal Hi-Hat',
    46: 3, #'Open Hi-Hat',
    54: 3, #'Tambourine',

    49: 4, #'Crash Cymbal 1',
    51: 4, #'Ride Cymbal 1',
    52: 4, #'Chinese Cymbal',
    53: 4, #'Ride Bell',
    55: 4, #'Splash Cymbal',
    56: 4, #'Cowbell',
    57: 4, #'Crash Cymbal 2',
    58: None, #'Vibraslap',
    59: 4, #'Ride Cymbal 2',

    27: None, #'High Q (GM2)',
    28: None, #'Slap (GM2)',
    29: None, #'Scratch Push (GM2)',
    30: None, #'Scratch Pull (GM2)',
    31: None, #'Sticks (GM2)',
    32: None, #'Square Click (GM2)',
    33: None, #'Metronome Click (GM2)',
    34: None, #'Metronome Bell (GM2)',
    60: None, #'Hi Bongo',
    61: None, #'Low Bongo',
    62: None, #'Mute Hi Conga',
    63: None, #'Open Hi Conga',
    64: None, #'Low Conga',
    65: None, #'High Timbale',
    66: None, #'Low Timbale',
    67: None, #'High Agogo',
    68: None, #'Low Agogo',
    69: None, #'Cabasa',
    70: None, #'Maracas',
    71: None, #'Short Whistle',
    72: None, #'Long Whistle',
    73: None, #'Short Guiro',
    74: None, #'Long Guiro',
    75: None, #'Claves',
    76: None, #'Hi Wood Block',
    77: None, #'Low Wood Block',
    78: None, #'Mute Cuica',
    79: None, #'Open Cuica',
    80: None, #'Mute Triangle',
    81: None, #'Open Triangle',
    82: None, #'Shaker (GM2)',
    83: None, #'Jingle Bell (GM2)',
    84: None, #'Belltree (GM2)',
    85: None, #'Castanets (GM2)',
    86: None, #'Mute Surdo (GM2)',
    87: None, #'Open Surdo (GM2)',

    # Manually retrieved from Roland TD-11
    22: 3, #'Unknown Hi Hat',
    26: 3, #'Unknown Hi Hat',
}

    

if __name__ == "__main__":
    # Declare the path to the dataset directory
    path = Path(__file__).resolve().parent.parent / "data" / "SADTP"

    print("\033[96m", f"Loading SADTP data into lists", "\033[0m", sep="")

    data, labels = [], []
    for track in path.iterdir():
        # Only directories are tracks
        if not track.is_dir() or track.name.startswith("."):
            continue

        audio_path = (track / "mix").with_suffix(".wav")
        midi_path = (track / "annotation").with_suffix(".mid")
            
        spectrogram = readAudio(audio_path)
        timesteps = spectrogram.shape[0]
        label = readMidi(midi_path, MIDI_MAPPING, timesteps, 5)

        partitions = timesteps // 400
        data += list(spectrogram.tensor_split(partitions, dim=0))
        labels += list(label.tensor_split(partitions, dim=0))

        # Temporary test
        test_loader = DataLoader(TensorDataset(spectrogram.tensor_split(partitions, dim=0), label.tensor_split(partitions, dim=0)), batch_size=16, num_workers=4, pin_memory=True)
        test_f1_micro, test_f1_macro, test_f1_class = evaluate_model(model, test_loader=test_loader, transforms=transforms, seed=123, device="cuda:0")
        print("\n", track.name)
        print(f"Micro F1: {test_f1_micro.item():.4f}")
        print(f"Macro F1: {test_f1_macro.item():.4f}")
        print(f"Class F1: {[f'{test_f1.item():.4f}' for test_f1 in test_f1_class]}")

    data, labels = torch.stack(data), torch.stack(labels)

    # Turn them into a Pytorch tensor dataset
    print("\033[96m", "     Creating tensor datasets", "\033[0m", sep="")
    dataset = TensorDataset(data, labels)

    new_path = (path / f"sadtp_test").with_suffix(".pt")
    print("\033[96m", "     Storing ", "\033[0m", f"{new_path.name}", "\033[96m", " to disk", "\033[0m", sep="")
    torch.save(dataset, new_path)

    print("\033[95m", "     Finished!", "\033[0m", sep="")

    # Load dataset and verify that everything is correct
    dataset = torch.load(new_path)
    print("\033[92m", "     Final dataset contains ", "\033[0m", len(dataset), "\033[92m", " entries", "\033[0m", sep="")
    print("\033[92m", "     Each entry has features of shape: ", "\033[0m", dataset[0][0].shape, "\033[92m", ", and labels of shape: ", "\033[0m", dataset[0][1].shape, sep="")

    # Verify that dataloaders work
    dataloader = DataLoader(torch.load(path / "sadtp_test.pt"), batch_size=16)
    num_batches, mean, std = len(dataloader), torch.zeros(1), torch.zeros(1)
    for i, (features, labels) in enumerate(dataloader):
        if i == 0:
            print("\033[92m", "Batched entry in dataloader has features of shape: ", "\033[0m", features.shape, "\033[92m", ", and labels of shape: ", "\033[0m", labels.shape, sep="")
        mean += torch.mean(features, dim=(0, 1, 2))
        std += torch.std(features, dim=(0, 1, 2))
