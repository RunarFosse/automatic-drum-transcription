import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

""" Run this file to turn a stored Tensorflow dataset into a stored PyTorch dataset """


if __name__ == "__main__":
    # Declare an argument parser for this file
    import argparse
    parser = argparse.ArgumentParser("datasets.py")
    parser.add_argument("dataset", help="The dataset to convert", type=str)
    args = parser.parse_args()

    # Get the path of the folder
    path = Path(__file__).resolve().parent / args.dataset

    print("\033[96m", "Loading TensorFlow dataset from disk", "\033[0m", sep="")

    # Load the dataset
    tf_dataset = tf.data.Dataset.load(str(path))

    print("\033[96m", "Loading data into lists", "\033[0m", sep="")

    # Store all features and labels
    features, labels = [], []
    for data, label in tf_dataset.as_numpy_iterator():
        features.append(data["x"])
        labels.append(label)
    features, labels = np.array(features), np.array(labels)

    print("\033[96m", "Creating tensor datasets", "\033[0m", sep="")

    # Turn them into a Pytorch tensor dataset
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))

    print("\033[96m", "Storing dataset to disk", "\033[0m", sep="")

    # And store the dataset to the disk under the first path
    torch.save(dataset, path.with_suffix(".pt"))

    print("\033[95m", "Finished!", "\033[0m", sep="")
    
    # Load dataset and verify that everything is correct
    dataset = torch.load(path.with_suffix(".pt"))
    print("\033[92m", "Final dataset contains ", "\033[0m", len(dataset), "\033[92m", " entries", "\033[0m", sep="")
    print("\033[92m", "Each entry has features of shape: ", "\033[0m", dataset[0][0].shape, "\033[92m", ", and labels of shape: ", "\033[0m", dataset[0][1].shape, sep="")

    # Verify that dataloaders work
    dataloader = DataLoader(dataset, batch_size=16)
    for features, labels in dataloader:
        print("\033[92m", "Batched entry in dataloader has features of shape: ", "\033[0m", features.shape, "\033[92m", ", and labels of shape: ", "\033[0m", labels.shape, sep="")
        break