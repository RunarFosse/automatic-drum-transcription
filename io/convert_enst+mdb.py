import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from load import readAudio, readAnnotations
from pathlib import Path

""" Run this file to turn ENST-Drums into a stored PyTorch dataset """

# Splits from ADTOF-github (https://github.com/MZehren/ADTOF/blob/master/adtof/ressources/splits.py)
# Originally from Vogl et al.
ENST_SPLITS = [
    [
        "107_minus-one_salsa_sticks",
        "108_minus-one_rock-60s_sticks",
        "109_minus-one_metal_sticks",
        "110_minus-one_musette_brushes",
        "111_minus-one_funky_rods",
        "112_minus-one_funk_rods",
        "113_minus-one_charleston_sticks",
        "114_minus-one_celtic-rock_brushes",
        "115_minus-one_bossa_brushes",
        "121_MIDI-minus-one_bigband_brushes",
        "123_MIDI-minus-one_blues-102_sticks",
        "125_MIDI-minus-one_country-120_brushes",
        "127_MIDI-minus-one_disco-108_sticks",
        "129_MIDI-minus-one_funk-101_sticks",
        "131_MIDI-minus-one_grunge_sticks",
        "133_MIDI-minus-one_nu-soul_sticks",
        "135_MIDI-minus-one_rock-113_sticks",
        "137_MIDI-minus-one_rock'n'roll-188_sticks",
        "139_MIDI-minus-one_soul-120-marvin-gaye_sticks",
        "141_MIDI-minus-one_soul-98_sticks",
        "143_MIDI-minus-one_fusion-125_sticks",
    ],
    [
        "115_minus-one_salsa_sticks",
        "116_minus-one_rock-60s_sticks",
        "117_minus-one_metal_sticks",
        "118_minus-one_musette_brushes",
        "119_minus-one_funky_sticks",
        "120_minus-one_funk_sticks",
        "121_minus-one_charleston_sticks",
        "122_minus-one_celtic-rock_sticks",
        "123_minus-one_celtic-rock-better-take_sticks",
        "124_minus-one_bossa_sticks",
        "130_MIDI-minus-one_bigband_sticks",
        "132_MIDI-minus-one_blues-102_sticks",
        "134_MIDI-minus-one_country-120_sticks",
        "136_MIDI-minus-one_disco-108_sticks",
        "138_MIDI-minus-one_funk-101_sticks",
        "140_MIDI-minus-one_grunge_sticks",
        "142_MIDI-minus-one_nu-soul_sticks",
        "144_MIDI-minus-one_rock-113_sticks",
        "146_MIDI-minus-one_rock'n'roll-188_sticks",
        "148_MIDI-minus-one_soul-120-marvin-gaye_sticks",
        "150_MIDI-minus-one_soul-98_sticks",
        "152_MIDI-minus-one_fusion-125_sticks",
    ],
    [
        "126_minus-one_salsa_sticks",
        "127_minus-one_rock-60s_sticks",
        "128_minus-one_metal_sticks",
        "129_minus-one_musette_sticks",
        "130_minus-one_funky_sticks",
        "131_minus-one_funk_sticks",
        "132_minus-one_charleston_sticks",
        "133_minus-one_celtic-rock_sticks",
        "134_minus-one_bossa_sticks",
        "140_MIDI-minus-one_bigband_sticks",
        "142_MIDI-minus-one_blues-102_sticks",
        "144_MIDI-minus-one_country-120_sticks",
        "146_MIDI-minus-one_disco-108_sticks",
        "148_MIDI-minus-one_funk-101_sticks",
        "150_MIDI-minus-one_grunge_sticks",
        "152_MIDI-minus-one_nu-soul_sticks",
        "154_MIDI-minus-one_rock-113_sticks",
        "156_MIDI-minus-one_rock'n'roll-188_sticks",
        "158_MIDI-minus-one_soul-120-marvin-gaye_sticks",
        "160_MIDI-minus-one_soul-98_sticks",
        "162_MIDI-minus-one_fusion-125_sticks",
    ],
]
MDB_SPLITS = [
    [
        "MusicDelta_Punk",
        "MusicDelta_CoolJazz",
        "MusicDelta_Disco",
        "MusicDelta_SwingJazz",
        "MusicDelta_Rockabilly",
        "MusicDelta_Gospel",
        "MusicDelta_BebopJazz",
    ],
    [
        "MusicDelta_FunkJazz",
        "MusicDelta_FreeJazz",
        "MusicDelta_Reggae",
        "MusicDelta_LatinJazz",
        "MusicDelta_Britpop",
        "MusicDelta_FusionJazz",
        "MusicDelta_Shadows",
        "MusicDelta_80sRock",
    ],
    [
        "MusicDelta_Beatles",
        "MusicDelta_Grunge",
        "MusicDelta_Zeppelin",
        "MusicDelta_ModalJazz",
        "MusicDelta_Country1",
        "MusicDelta_SpeedMetal",
        "MusicDelta_Rock",
        "MusicDelta_Hendrix",
    ],
]

# 5-drum mapping for ENST and MDB, from ADTOF
ENST_MAPPING = {
    "bd": 0,

    "sd": 1,
    "sd-": 1,
    "sweep": 1,
    "rs": 1,
    "sticks": 1,
    "cs": 1,

    "mt": 2,
    "mtr": 2,
    "lmt": 2,
    "lt": 2,
    "ltr": 2,
    "lft": 2,

    "chh": 3,
    "ohh": 3,

    "cr": 4,
    "spl": 4,
    "ch": 4,
    "rc": 4,
    "c": 4,
    "cb": 4,
}

MDB_MAPPING = {
    "KD": 0, #'kick drum'

    "SD": 1, #'snare drum',
    "SDB": 1, #'snare drum: brush',
    "SDD": 1, #'snare drum: drag',
    "SDF": 1, #'snare drum: flam',
    "SDG": 1, #'snare drum: ghost note',
    "SDNS": 1, #'snare drum: no snare',
    "SST": 1, #'side stick',

    "HIT": 2, #'high tom',
    "MHT": 2, #'high-mid tom',
    "HFT": 2, #'high floor tom',
    "LFT": 2, #'low floor tom',

    "CHH": 3, #'hi-hat: closed',
    "OHH": 3, #'hi-hat: open',
    "PHH": 3, #'hi-hat: pedal',
    "TMB": 3, #'tambourine',

    "RDC": 4, #'ride cymbal',
    "RDB": 4, #'ride cymbal: bell',
    "CRC": 4, #'crash cymbal',
    "CHC": 4, #'china cymbal',
    "SPC": 4, #'splash cymbal',
}

# Set a seed for predictable splitting
seed = 100

if __name__ == "__main__":
    # Declare the path to the dataset directory
    path = Path(__file__).resolve().parent.parent / "data" / "ENST+MDB"

    print("\033[96m", "Loading data into lists", "\033[0m", sep="")

    data, labels = [], []
    for drummer in range(3):
        for piece in ENST_SPLITS[drummer]:
            audio_path = (path / "ENST-drums-public" / f"drummer_{drummer+1}" / "audio" / "wet_mix" / piece).with_suffix(".wav")
            accompaniment_path = (path / "ENST-drums-public" / f"drummer_{drummer+1}" / "audio" / "accompaniment" / piece).with_suffix(".wav")
            annotation_path = (path / "ENST-drums-public" / f"drummer_{drummer+1}" / "annotation" / piece).with_suffix(".txt")

            spectrogram = readAudio(audio_path, accompaniment_path)
            timesteps = spectrogram.shape[0]
            label = readAnnotations(annotation_path, ENST_MAPPING, timesteps, 5)

            partitions = timesteps // 400
            data += list(spectrogram.tensor_split(partitions, dim=0))
            labels += list(label.tensor_split(partitions, dim=0))
        for piece in MDB_SPLITS[drummer]:
            audio_path = (path / "MDBDrums-master" / "MDB Drums" / "audio" / "full_mix" / f"{piece}_MIX").with_suffix(".wav")
            annotation_path = (path / "MDBDrums-master" / "MDB Drums" / "annotations" / "subclass" / f"{piece}_subclass").with_suffix(".txt")
            
            spectrogram = readAudio(audio_path)
            timesteps = spectrogram.shape[0]
            label = readAnnotations(annotation_path, MDB_MAPPING, timesteps, 5)

            partitions = timesteps // 400
            data += list(spectrogram.tensor_split(partitions, dim=0))
            labels += list(label.tensor_split(partitions, dim=0))

    data, labels = torch.stack(data), torch.stack(labels)
    
    # Turn them into a Pytorch tensor dataset
    print("\033[96m", "Creating tensor datasets", "\033[0m", sep="")
    dataset = TensorDataset(data, labels)

    # Split them into Train/Validation/Test sets TODO! WIP: MAY HAVE DATALEAKAGE (same song might appear in several splits) TODO!
    print("\033[96m", "Creating train/validation/test splits", "\033[0m", sep="")
    torch.manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, [0.7, 0.15, 0.15])

    # Verify split sizes
    print("\033[96m", "Train size: ", "\033[0m", len(train_dataset.indices), "\033[0m", sep="")
    print("\033[96m", "Validation size: ", "\033[0m", len(validation_dataset.indices), "\033[0m", sep="")
    print("\033[96m", "Test size: ", "\033[0m", len(test_dataset.indices), "\033[0m", sep="")

    # Verify indices don't overlap
    train_collisions = len(set(train_dataset.indices).intersection(set(validation_dataset.indices))) + len(set(train_dataset.indices).intersection(set(test_dataset.indices)))
    validation_collisions = len(set(validation_dataset.indices).intersection(set(test_dataset.indices)))
    print("\033[96m", "Number of collisions: ", "\033[0m", train_collisions + validation_collisions, "\033[0m", sep="")

    # Store every split
    for split, dataset in [("train", train_dataset), ("validation", validation_dataset), ("test", test_dataset)]:
        # And store the dataset to the disk under the first path
        new_path = (path / f"enst+mdb_{split}").with_suffix(".pt")
        print("\033[96m", "     Storing ", "\033[0m", f"{new_path.name}", "\033[96m", " to disk", "\033[0m", sep="")
        torch.save(dataset, new_path)

        print("\033[95m", "     Finished!", "\033[0m", sep="")
    
        # Load dataset and verify that everything is correct
        dataset = torch.load(new_path)
        print("\033[92m", "     Final dataset contains ", "\033[0m", len(dataset), "\033[92m", " entries", "\033[0m", sep="")
        print("\033[92m", "     Each entry has features of shape: ", "\033[0m", dataset[0][0].shape, "\033[92m", ", and labels of shape: ", "\033[0m", dataset[0][1].shape, sep="")

    # Verify that dataloaders work
    dataloader = DataLoader(train_dataset, batch_size=16)
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