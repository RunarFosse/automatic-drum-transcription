import torch
import numpy as np
import tensorflow as tf
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
from pathlib import Path

""" Run this file to turn a stored Tensorflow dataset into a stored PyTorch dataset """


class TensorFlowDatasetIterable(IterableDataset):
    """ Load a TensorFlow dataset as a PyTorch iterable dataset

    Arguments:
        path (string): Path to the given dataset folder
        transform (callable, optional): Optional transform to be applied to a sample
    """
    def __init__(self, tf_dataset : tf.data.Dataset, transform = None):
        # Load the dataset using TensorFlow
        self._tf_dataset = tf_dataset
        self._transform = transform
    
    def __len__(self):
        return self._tf_dataset.cardinality().numpy()
    
    def __iter__(self):
        # Iterate the dataset
        for data, label in self._tf_dataset.as_numpy_iterator():
            features = torch.tensor(data["x"])
            label = torch.tensor(label)

            # Transform the features
            if self._transform:
                features = self._transform(features)
            
            # And permute the dimensions
            features = features.permute((0, 3, 1, 2))

            # And yield the datapoints
            yield features, label


def ADTOF_load(path: Path, batch_size = 1, shuffle = False, transform=None, seed=None) -> DataLoader:
    """ Load a ADTOF dataset as a PyTorch DataLoader """

    # Load the dataset using TensorFlow on CPU
    with tf.device('/device:cpu:0'):
        tf_dataset = tf.data.Dataset.load(str(path))

        # Let TensorFlow handle batching and shuffling
        tf_dataset = tf_dataset.batch(batch_size = batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        if shuffle:
            tf_dataset = tf_dataset.shuffle(buffer_size = batch_size * 50, seed = seed)

        # Wrap the dataset as a PyTorch iterable dataset
        dataset = TensorFlowDatasetIterable(tf_dataset, transform = transform)

    # And return it as a PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size = None)
    return dataloader



if __name__ == "__main__":
    # Declare an argument parser for this file
    import argparse
    parser = argparse.ArgumentParser("datasets.py")
    parser.add_argument("dataset", help="The dataset to convert", type=str)
    args = parser.parse_args()

    # Get the path of the folder
    path = Path(__file__).resolve().parent / args.dataset

    print("\033[96m", "Loading TensorFlow dataset from disk", "\033[0m")

    # Load the dataset
    tf_dataset = tf.data.Dataset.load(str(path))

    print("\033[96m", "Loading data into lists", "\033[0m")

    # Store all features and labels
    features, labels = [], []
    for data, label in tf_dataset.as_numpy_iterator():
        features.append(data["x"])
        labels.append(label)
    features, labels = np.array(features), np.array(labels)

    print("\033[96m", "Creating tensor datasets", "\033[0m")

    # Turn them into a Pytorch tensor dataset
    dataset = TensorDataset(torch.tensor(features), torch.tensor(labels))

    print("\033[96m", "Storing dataset to disk", "\033[0m")

    # And store the dataset to the disk under the first path
    torch.save(dataset, path.with_suffix(".pt"))

    print("\033[92m", "Finished!", "\033[0m")
    
    # Load dataset and verify that everything is correct
    dataset = torch.load(path.with_suffix(".pt"))
    print("Final dataset contains", "\033[95m", len(dataset), "\033[0m", "entries")
    print("Each entry has features of shape:", "\033[95m", dataset[0][0].shape, "\033[0m", "and labels of shape", "\033[95m", dataset[0][1].shape, "\033[0m")

    # Verify that dataloaders work
    dataloader = DataLoader(dataset, batch_size=16)
    for features, labels in dataloader:
        print("Batched entry in dataloader has features of shape:", "\033[95m", features.shape, "\033[0m", "and labels of shape", "\033[95m", labels.shape, "\033[0m")
        break