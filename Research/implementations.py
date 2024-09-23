import numpy as np

def frames_to_time(frames: np.ndarray, hop_length: int, sr: int = 22050) -> np.ndarray:
    """
    For a given array of frames, calculate frame onset times in seconds, given hop length :param:`hop_length and sampling rate :param:`sr`
    """
    time = []

    total_frames = frames.size
    for frame in range(total_frames):
        onset_sample = frame * hop_length
        time.append(onset_sample / sr)
    
    return np.array(time)

def amplitude_envelope(waveform: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    """
    For a given waveform, calculate the amplitude envelope over frames of size :param:`frame_size` at hop length :param:`hop_length`
    """
    amplitudes = []

    total_samples = waveform.size
    for start_sample in range(0, total_samples, hop_length):
        end_sample = min(start_sample + frame_size, total_samples)

        maximum_amplitude = np.max(waveform[start_sample:end_sample])
        amplitudes.append(maximum_amplitude)
    
    return np.array(amplitudes)

def rms(waveform: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    """
    For a given waveform, calculate the root-mean-squared energy over frames of size :param:`frame_size` at hop length :param:`hop_length`
    """
    energies = []

    total_samples = waveform.size
    for start_sample in range(0, total_samples, hop_length):
        end_sample = min(start_sample + frame_size, total_samples)

        energy = np.sqrt(np.mean(np.power(waveform[start_sample:end_sample], 2.0)))
        energies.append(energy)
    
    return np.array(energies)

def zcr(waveform: np.ndarray, frame_size: int, hop_length: int) -> np.ndarray:
    """
    For a given waveform, calculate the zero-crossing rate over frames of size :param:`frame_size` at hop length :param:`hop_length`
    """
    rates = []

    total_samples = waveform.size
    for start_sample in range(0, total_samples, hop_length):
        end_sample = min(start_sample + frame_size, total_samples)

        rate = 0.0
        for sample in range(start_sample, end_sample - 1):
            rate += np.abs(np.sign(waveform[sample]) - np.sign(waveform[sample + 1]))
        rates.append(rate / 2.0)
    
    return np.array(rates)

def dft(signal: np.ndarray) -> np.ndarray:
    """
    For a given signal calculate the discrete fourier transform. Output is a numpy array of complex values
    """
    fourier_transform = []

    N = signal.size
    for frequency in range(N):
        transform = 0
        for n in range(N):
            transform += signal[n] * np.exp(-1j * 2 * np.pi * frequency / N * n)
        
        fourier_transform.append(transform)

    return np.array(fourier_transform)

def fft(signal: np.ndarray) -> np.ndarray:
    """
    For a given signal calculate the discrete fourier transform approximation using the FFT algorithm. Output is a numpy array of complex values
    Inputs of length not on the form 2^n will be 0 padded.
    """

    n = signal.size
    if n == 1:
        return signal

    # If n is not a power of 2, right pad with 0s
    if n.bit_count() > 1:
        padding = int(np.power(2, np.ceil(np.log2(n))) - n)
        signal = np.concatenate((signal, [0] * padding))
        n += padding

    root_of_unity = np.exp(2 * np.pi * 1j / n)

    signal_even, signal_odd = np.array([signal[i] for i in range(0, n, 2)]), np.array([signal[i] for i in range(1, n, 2)])
    transform_even, transform_odd = fft(signal_even), fft(signal_odd)

    transform = [0] * n
    for j in range(n // 2):
        transform[j] = transform_even[j] + np.power(root_of_unity, j) * transform_odd[j]
        transform[j + n // 2] = transform_even[j] - np.power(root_of_unity, j) * transform_odd[j]

    return np.array(transform)

def hann_window(frame_size: int) -> np.ndarray:
    """
    Compute the Hann window for a frame of size :param:`frame_size`
    """
    window = np.zeros(frame_size)
    for i in range(frame_size):
        window[i] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (i+1) / (frame_size - 1.0)))

    return window

def stft(signal: np.ndarray, frame_size: int, hop_length: int, center: bool = True) -> np.ndarray:
    """
    For a given signal, frame size and hop length, compute the Short-time Fourier transform
    """
    window = hann_window(frame_size)

    total_samples = signal.size

    total_frames = total_samples // hop_length + 1
    total_frequency_bins = frame_size // 2 + 1

    spectrogram = np.zeros((total_frequency_bins, total_frames), dtype=np.complex_)
    for i in range(total_frames):
        if center:
            start_sample = i * hop_length - frame_size // 2
        else:
            start_sample = i * hop_length
        end_sample = start_sample + frame_size

        if start_sample < 0:
            frame = np.append(np.zeros(np.abs(start_sample)), signal[:end_sample])
        elif end_sample > total_samples:
            frame = np.append(signal[start_sample:], np.zeros(end_sample - total_samples))
        else:
            frame = signal[start_sample:end_sample]

        frame_fft = np.fft.fft(frame * window)
        spectrogram[:, i] = frame_fft[:total_frequency_bins]

    return spectrogram