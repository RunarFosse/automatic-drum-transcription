import tensorflow as tf
import torch
from torch.utils.data import Dataset


class ADTDataset(Dataset):
    """ Generic dataset class for Automatic drum transcription tasks """

    """
    Arguments:
        path (string): Path to the given dataset folder
        transform (callable, optional): Optional transform to be applied to a sample
    """
    def __init__(self, path: str, transform = None):
        # Load the dataset using tensorflow (as this is what they are stored as)
        self._tf_dataset = tf.data.Dataset.load(path)
        
    def __len__(self):
        # Extract the length of the dataset by using tensorflows data.experimental.cardinality function
        return tf.data.experimental.cardinality(self._tf_dataset).numpy()
    
    def __getitem__(self, index: int):
        # Select the wanted dataset item
        item = next(self._tf_dataset.skip(index).as_numpy_iterator())

        # Convert to PyTorch tensors and return
        features = torch.tensor(item[0]["x"])
        label = torch.tensor(item[1])
        return features, label