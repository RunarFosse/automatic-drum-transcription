import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize
from load import readAudio, readAnnotations
from pathlib import Path
import pandas as pd
import mido

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
    59: 4, #'Ride Cymbal 2',
}

if __name__ == "__main__":
    # Declare an argument parser for this file
    import argparse
    parser = argparse.ArgumentParser("convert_enst.py")
    parser.add_argument("path", help="The path to the E-GMD dataset", type=str)
    args = parser.parse_args()

    path = Path(__file__).resolve().parent.parent / args.path

    # Load the CSV
    csv = pd.read_csv(path / "e-gmd-v1.0.0.csv")

    # Store label and frame indices in lists
    frame_indices = []
    label_indices = []

    # Split into Train, Validation, Test
    for split in ["train", "validation", "test"]:
        df = csv[csv["split"] == split]
        for row in df.iterrows():
            values = row[1]
            audio_path = path / values["audio_filename"]
            midi_path = path / values["midi_filename"]
            
            #midi = pretty_midi.PrettyMIDI(midi_path.as_posix(), initial_tempo=int(values["bpm"]))
            midi = mido.MidiFile(midi_path)
            
            # Iterate the MIDI file
            for msg in midi.tracks[1]:
                pass