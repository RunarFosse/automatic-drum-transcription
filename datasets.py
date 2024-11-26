import tensorflow as tf
import torch
from torch.utils.data import IterableDataset, DataLoader
from pathlib import Path


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
    
    def __iter__(self):
        # Iterate the dataset
        for data, label in self._tf_dataset:
            features = torch.tensor(data["x"].numpy())
            label = torch.tensor(label.numpy())

            # Transform the features
            if self._transform:
                features = self._transform(features)
            
            # And permute the dimensions
            features = features.permute((0, 3, 1, 2))

            # And yield the datapoints
            yield features, label


def ADTOF_load(path: Path, batch_size = 1, shuffle = False, transform=None, seed=None) -> DataLoader:
    """ Load a ADTOF dataset as a PyTorch DataLoader """

    with tf.device("/cpu:0"):
        # Load the dataset using TensorFlow
        tf_dataset = tf.data.Dataset.load(str(path))

        # Let TensorFlow handle batching and shuffling
        tf_dataset = tf_dataset.shuffle(buffer_size = batch_size * 25, seed = seed)
        tf_dataset = tf_dataset.batch(batch_size = batch_size)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)

        # Wrap the dataset as a PyTorch iterable dataset
        dataset = TensorFlowDatasetIterable(tf_dataset, transform = transform)

    # And return it as a PyTorch DataLoader
    dataloader = DataLoader(dataset, batch_size = None)
    return dataloader