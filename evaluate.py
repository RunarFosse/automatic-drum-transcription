import torch
import torch.nn.functional as F


def compute_peaks(activations: torch.Tensor, m: int = 2, o: int = 2, w: int = 2, delta: float = 0.1) -> torch.Tensor:
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


def compute_predictions(activations: torch.Tensor, annotations: torch.Tensor, w: int = 5) -> torch.Tensor:
    """ Compute the F-measure of a given an activation sequence """

    n_batches, n_labels = activations.shape[0:2]

    # Store predictions per class, [True Positives, False Positives, False Negatives]
    predictions = torch.zeros(size=(n_labels, 3))

    for batch in range(n_batches):
        for label in range(n_labels):
            activation_onsets = torch.where(activations[batch][label] > 0.5)[0]
            annotation_onsets = torch.where(annotations[batch][label] > 0.5)[0]
    
            activation_pointer, annotation_pointer = 0, 0
            while activation_pointer < activation_onsets.shape[0] and annotation_pointer < annotation_onsets.shape[0]:
                # Check for True Positive
                if abs(activation_onsets[activation_pointer] - annotation_onsets[annotation_pointer]) <= w:
                    predictions[label, 0] += 1
                    activation_pointer += 1
                    annotation_pointer += 1

                # Check for False Positive
                elif activation_onsets[activation_pointer] < annotation_onsets[annotation_pointer]:
                    predictions[label, 1] += 1
                    activation_pointer += 1

                # Check for False Negative
                else:
                    predictions[label, 2] += 1
                    annotation_pointer += 1
        
            # Count remaining False Positives
            if activation_pointer < activation_onsets.shape[0]:
                predictions[label, 1] += activation_onsets.shape[0] - activation_pointer
    
            # Count remaining False Negatives
            if annotation_pointer < annotation_onsets.shape[0]:
                predictions[label, 2] += annotation_onsets.shape[0] - annotation_pointer
    
    return predictions


def f_measure(predictions: torch.Tensor):
    """ Given computed predictions (True Positives, False Positives, False Negatives), compute the F-measure, precision and recall, global and per class """

    global_precision = torch.sum(predictions[:, 0]) / torch.sum(predictions[:, 0] + predictions[:, 1])
    class_precision = predictions[:, 0] / (predictions[:, 0] + predictions[:, 1])

    global_recall = torch.sum(predictions[:, 0]) / torch.sum(predictions[:, 0] + predictions[:, 2])
    class_recall = predictions[:, 0] / (predictions[:, 0] + predictions[:, 2])

    global_f1 = 2.0 * (global_precision * global_recall) / (global_precision + global_recall)
    class_f1 = 2.0 * (class_precision * class_recall) / (class_precision + class_recall)

    return global_f1, class_f1

if __name__ == "__main__":
    y_pred = torch.tensor([[
        [0.1, 0.3, 0.7, 0.4, 0.5, 0.9, 0.2, 0.8, 0.2, 0.4, 0.6, 0.3],
        [0.2, 0.6, 0.1, 0.4, 0.8, 0.3, 0.7, 0.2, 0.5, 0.9, 0.2, 0.1]
    ],
    [
        [0.0, 0.1, 0.7, 0.2, 0.3, 0.7, 0.9, 0.8, 0.7, 0.7, 0.7, 0.9],
        [0.0, 0.2, 0.3, 0.2, 0.1, 0.0, 0.0, 0.9, 0.0, 0.9, 0.0, 0.9]
    ]])

    annotations = torch.tensor([[
        [0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5],
        [0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ],
    [
        [0.5, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5],
        [0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.5, 1.0, 0.5, 1.0, 0.5, 1.0]
    ]])

    prediction = compute_peaks(y_pred)
    print("Predictions:\n", prediction.round())
    print("Annotations:\n", annotations.round())
    predictions = compute_predictions(prediction, annotations, w=2)
    f_global, f_class = f_measure(predictions)
    print("Global F1-score:", f_global)
    print("Classwise F1-score:", f_class)