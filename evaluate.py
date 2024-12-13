import torch
import torch.nn.functional as F

def compute_peaks(activations: torch.Tensor, m: int = 2, o: int = 2, w: int = 2, delta: float = 0.1):
    """ Compute the peaks of a given activation time series using Vogl's peak picking algorithm """

    # First pad the data to handle boundary conditions
    padded_activations = F.pad(activations, (m, m), value = 0)

    # Then compute sliding windows over each of the activations
    windows = padded_activations.unfold(dimension = -1, size = 2*m + 1, step = 1)

    # Then compute peaks using Vogl's max and mean criteria
    is_peak = (activations == torch.amax(windows, dim=-1)) & (activations >= torch.mean(windows, dim=-1) + delta)
    
    # Enforce minimum distance between peaks
    n_batches, n_labels = activations.shape[0:2]
    filtered_peaks = []
    for batch in range(n_batches):
        for label in range(n_labels):
            peak_indices = torch.where(is_peak[batch][label])[0]
            last_peak = -(w - 1)
            for peak in peak_indices:
                if peak - last_peak > w:
                    filtered_peaks.append([batch, label, peak])
                    last_peak = peak
    
    # Return the output masked by the peaks
    mask = torch.zeros_like(activations)
    mask[*torch.tensor(filtered_peaks).T] = 1

    return activations * mask

if __name__ == "__main__":
    batch_data = torch.tensor([[
        [0.1, 0.3, 0.7, 0.4, 0.5, 0.9, 0.2, 0.8, 0.6, 0.4, 0.9, 0.3],
        [0.2, 0.6, 0.1, 0.4, 0.8, 0.3, 0.7, 0.2, 0.5, 0.9, 0.2, 0.1]
    ],
    [
        [0.0, 0.1, 0.7, 0.2, 0.3, 0.7, 0.9, 0.8, 0.7, 0.7, 0.7, 0.9],
        [0.0, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9]
    ]])

    print(compute_peaks(batch_data))