import matplotlib.pyplot as plt
import numpy as np
import librosa
from typing import Tuple

def plot_waveform(waveform: np.ndarray, sr: int = 22050, title: str = "") -> None:
    librosa.display.waveshow(waveform, sr=sr, alpha=0.5)
    plt.title(title)
    plt.ylabel("Relative intensity")