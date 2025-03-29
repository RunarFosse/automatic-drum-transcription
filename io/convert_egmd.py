import torch
from torch.utils.data import TensorDataset, DataLoader
from load import readAudio, readMidi
from pathlib import Path
import pandas as pd

""" Run this file to turn E-GMD into a stored PyTorch dataset """

# 5-drum mapping for EGMD, from "https://github.com/khiner/DrumClassification/blob/main/create_label_mapping.py"
EGMD_MAPPING = {
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

    22: None, #Unknown Mapping - Not Percussion
    26: None, #Unknown Mapping - Not Percussion
}

if __name__ == "__main__":
    # Declare the path to the dataset directory
    path = Path(__file__).resolve().parent.parent / "data" / "e-gmd-v1.0.0"

    # Load the CSV
    csv = pd.read_csv(path / "e-gmd-v1.0.0.csv")

    # Split into Train, Validation, Test
    for split in ["train", "validation", "test"]:

        print("\033[96m", f"Loading {split} data into lists", "\033[0m", sep="")

        data, labels = [], []
        df = csv[csv["split"] == split]
        for row in df.iterrows():
            values = row[1]
            audio_path = path / values["audio_filename"]
            midi_path = path / values["midi_filename"]
            
            spectrogram = readAudio(audio_path)
            timesteps = spectrogram.shape[0]
            label = readMidi(midi_path, EGMD_MAPPING, timesteps, 5)

            partitions = timesteps // 400
            data += list(spectrogram.tensor_split(partitions, dim=0))
            labels += list(label.tensor_split(partitions, dim=0))

            if spectrogram.shape[0] % 400 != 0:
                print(split.capitalize() + ":", audio_path.stem)
                print("      ", spectrogram.shape)
                print("      ", label.shape)
                print(partitions)
        
        data, labels = torch.stack(data), torch.stack(labels)
    
        # Turn them into a Pytorch tensor dataset
        print("\033[96m", "     Creating tensor datasets", "\033[0m", sep="")
        dataset = TensorDataset(data, labels)

        new_path = (path / f"egmd_{split}").with_suffix(".pt")
        print("\033[96m", "     Storing ", "\033[0m", f"{new_path.name}", "\033[96m", " to disk", "\033[0m", sep="")
        torch.save(dataset, new_path)

        print("\033[95m", "     Finished!", "\033[0m", sep="")
    
        # Load dataset and verify that everything is correct
        dataset = torch.load(new_path)
        print("\033[92m", "     Final dataset contains ", "\033[0m", len(dataset), "\033[92m", " entries", "\033[0m", sep="")
        print("\033[92m", "     Each entry has features of shape: ", "\033[0m", dataset[0][0].shape, "\033[92m", ", and labels of shape: ", "\033[0m", dataset[0][1].shape, sep="")

    # Verify that dataloaders work
    dataloader = DataLoader(torch.load(path / "egmd_train.pt"), batch_size=16)
    num_batches, mean, std = len(dataloader), torch.zeros(1), torch.zeros(1)
    for i, (features, labels) in enumerate(dataloader):
        if i == 0:
            print("\033[92m", "Batched entry in dataloader has features of shape: ", "\033[0m", features.shape, "\033[92m", ", and labels of shape: ", "\033[0m", labels.shape, sep="")
        mean += torch.mean(features, dim=(0, 1, 2))
        std += torch.std(features, dim=(0, 1, 2))

    # Compute mean and std of dataset
    mean /= num_batches
    std /= num_batches
    print("\033[92m", "Training dataset has mean of: ", "\033[0m", mean, "\033[92m", ", and std of: ", "\033[0m", std, sep="")
        

        