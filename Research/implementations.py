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
