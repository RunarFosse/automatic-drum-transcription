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
    if not transform.size:
        plt.text(x=0.0, y=0.0, s="Fourier transform not computed", fontsize="xx-large", fontweight="bold", horizontalalignment="center")
        plot_nyquist = False

    plt.plot(np.linspace(0, sr, len(transform)), np.abs(transform), label="Fourier Transform")

    if plot_nyquist:
        plt.axvline(x=sr / 2.0, c="r", linestyle="--", label="Nyquist Frequency")
    
    if transform.size:
        plt.legend()

    plt.ylabel("Magnitude")
    plt.xlabel("Frequency (Hz)")
    plt.title(title)

def plot_spectrogram(spectrogram: np.ndarray, hop_length: int, sr: int = 22050, title: str = "", y_axis: str = "linear", color_unit: str = "", ylabel: str = "") -> None:
    """Plot a spectrogram"""
    librosa.display.specshow(spectrogram, sr=sr, hop_length=hop_length, x_axis="time", y_axis=y_axis)
    plt.colorbar(format="%+2.f " + color_unit)

    plt.title(title)
    if ylabel:
        plt.ylabel(ylabel)

def plot_mel_filter_banks(filter_banks: np.ndarray, n_fft: int, title: str = "") -> None:
    frequencies = np.arange(0, n_fft // 2 + 1)

    for i, filter_bank in enumerate(filter_banks):
        plt.plot(frequencies, filter_bank, label=f"Filter bank {i + 1}")

    plt.title(title)
    plt.xlabel("Frequencies")
    plt.ylabel("Weight value")
    plt.legend()