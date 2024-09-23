import matplotlib.pyplot as plt
import numpy as np
import librosa
from typing import Tuple

def plot_waveform(waveform: np.ndarray, sr: int = 22050, title: str = "") -> None:
    """Plot a given waveform"""
    librosa.display.waveshow(waveform, sr=sr, alpha=0.5)
    plt.ylabel("Relative intensity")
    plt.title(title)

def plot_fourier_transform(transform: np.ndarray, sr: int = 22050, title: str = "", plot_nyquist: bool = False) -> None:
    """Plot a fourier transform"""
    plt.plot(np.linspace(0, sr, len(transform)), np.abs(transform), label="Fourier Transform")

    if plot_nyquist:
        plt.axvline(x=sr / 2.0, c="r", linestyle="--", label="Nyquist Frequency")
    
    plt.ylabel("Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.title(title)
    plt.legend()

def plot_spectrogram(spectrogram: np.ndarray, hop_length: int, sr: int = 22050, title: str = "", y_axis: str = "linear", color_unit: str = "") -> None:
    """Plot a spectrogram"""
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f " + color_unit)

    plt.title(title)