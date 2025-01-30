import torch
import torch.nn.functional as F


def compute_peaks(activations: torch.Tensor, m: int = 2, o: int = 2, w: int = 2, delta: float = 0.1) -> torch.Tensor:
    """ Compute the peaks of a given activation time series using Vogl's peak picking algorithm """

    # First pad the data to handle boundary conditions
    padded_activations = F.pad(activations, (0, 0, m, m), value=0)

    # Then compute sliding windows over each of the activations
    windows = padded_activations.unfold(dimension=-2, size=2*m + 1, step=1)

    # Then compute peaks using Vogl's max and mean criteria
    window_max = torch.amax(windows, dim=-1)
    window_mean = torch.mean(windows, dim=-1)
    is_peak = (activations == window_max) & (activations >= window_mean + delta)

    # Compute peak indices
    peak_indices = torch.where(is_peak)

    # Perform sequential logic below on CPU
    peak_indices = [indices.cpu() for indices in peak_indices]
    
    # Enforce minimum distance between peaks
    n_batches, n_labels = activations.shape[0], activations.shape[2]
    filtered_peaks = []
    for batch in range(n_batches):
        for label in range(n_labels):
            peaks = peak_indices[1][(peak_indices[0] == batch) & (peak_indices[2] == label)]

            last_peak = -(w - 1)
            for peak in peaks:
                if peak - last_peak > w:
                    filtered_peaks.append((batch, peak.item(), label))
                    last_peak = peak
    
    # If there are no peaks, return zero tensor
    if not filtered_peaks:
        return torch.zeros_like(activations)
        
    # Use a sparse tensor as peaks are generally rare
    batch_indices, peak_indices, label_indices = zip(*filtered_peaks)
    mask = torch.sparse_coo_tensor(
        torch.tensor([batch_indices, peak_indices, label_indices]),
        torch.ones(len(filtered_peaks)),
        activations.shape
    ).to(device=activations.device).to_dense()

    # Return the output masked by the peaks
    return activations * mask


def compute_predictions(activations: torch.Tensor, annotations: torch.Tensor, w: int = 5) -> torch.Tensor:
    """ Compute the F-measure of a given an activation sequence """

    n_batches, n_labels = activations.shape[0], activations.shape[2]

    # Store predictions per class, [True Positives, False Positives, False Negatives]
    predictions = torch.zeros(size=(n_labels, 3))

    # Compute onsets for both activations and annotations
    activation_onsets = torch.where(activations > 0.5)
    annotation_onsets = torch.where(annotations > 0.5)
    
    # Perform sequential logic below on CPU
    activation_onsets = [onsets.cpu() for onsets in activation_onsets]
    annotation_onsets = [onsets.cpu() for onsets in annotation_onsets]

    for batch in range(n_batches):
        for label in range(n_labels):
            # Extract onsets for current label and batch
            batch_activation_onsets = activation_onsets[1][(activation_onsets[0] == batch) & (activation_onsets[2] == label)]
            batch_annotation_onsets = annotation_onsets[1][(annotation_onsets[0] == batch) & (annotation_onsets[2] == label)]
    
            activation_pointer, annotation_pointer = 0, 0
            while activation_pointer < len(batch_activation_onsets) and annotation_pointer < len(batch_annotation_onsets):
                # Check for True Positive
                if abs(batch_activation_onsets[activation_pointer] - batch_annotation_onsets[annotation_pointer]) <= w:
                    predictions[label, 0] += 1
                    activation_pointer += 1
                    annotation_pointer += 1

                # Check for False Positive
                elif batch_activation_onsets[activation_pointer] < batch_annotation_onsets[annotation_pointer]:
                    predictions[label, 1] += 1
                    activation_pointer += 1

                # Check for False Negative
                else:
                    predictions[label, 2] += 1
                    annotation_pointer += 1
        
            # Count remaining False Positives and False Negatives
            predictions[label, 1] += len(batch_activation_onsets) - activation_pointer
            predictions[label, 2] += len(batch_annotation_onsets) - annotation_pointer
    
    return predictions


def f_measure(predictions: torch.Tensor):
    """ Given computed predictions (True Positives, False Positives, False Negatives), compute the F-measure, precision and recall, global and per class """

    global_precision = torch.sum(predictions[:, 0]) / torch.sum(predictions[:, 0] + predictions[:, 1])
    class_precision = predictions[:, 0] / (predictions[:, 0] + predictions[:, 1])

    global_recall = torch.sum(predictions[:, 0]) / torch.sum(predictions[:, 0] + predictions[:, 2])
    class_recall = predictions[:, 0] / (predictions[:, 0] + predictions[:, 2])

    global_f1 = 2.0 * (global_precision * global_recall) / (global_precision + global_recall)
    class_f1 = 2.0 * (class_precision * class_recall) / (class_precision + class_recall)

    # Replace any NaN resulting from zero-division, with zeros
    global_f1 = global_f1.nan_to_num(nan=0.0)
    class_f1 = class_f1.nan_to_num(nan=0.0)

    return global_f1, class_f1

if __name__ == "__main__":
    y_pred = torch.tensor([[
        [0.1, 0.3, 0.7, 0.4, 0.5, 0.9, 0.2, 0.8, 0.2, 0.4, 0.6, 0.3],
        [0.2, 0.6, 0.1, 0.4, 0.8, 0.3, 0.7, 0.2, 0.5, 0.9, 0.2, 0.1],
        [0.1, 0.2, 0.3, 0.4, 0.7, 0.4, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0]
    ],
    [
        [0.0, 0.1, 0.7, 0.2, 0.3, 0.7, 0.9, 0.8, 0.7, 0.7, 0.7, 0.9],
        [0.0, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.1, 0.0, 0.0]
    ]]).permute((0, 2, 1))

    annotations = torch.tensor([[
        [0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5],
        [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5],
        [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ]]).permute((0, 2, 1))

    prediction = compute_peaks(y_pred)
    print("Predictions:\n", prediction.round())
    print("Annotations:\n", annotations.round())
    predictions = compute_predictions(prediction, annotations, w=2)
    print("Predictions:", predictions)
    f_global, f_class = f_measure(predictions)
    print("Global F1-score:", f_global)
    print("Classwise F1-score:", f_class)