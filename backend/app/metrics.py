import numpy as np


def binary_mask_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict[str, float]:
    if prediction.ndim != 2 or ground_truth.ndim != 2:
        raise ValueError("Prediction and ground-truth masks must be two-dimensional")
    if prediction.shape != ground_truth.shape:
        raise ValueError(
            f"Mask dimensions do not match: prediction {prediction.shape}, "
            f"ground truth {ground_truth.shape}"
        )

    predicted = prediction.astype(bool)
    target = ground_truth.astype(bool)
    intersection = int(np.count_nonzero(predicted & target))
    union = int(np.count_nonzero(predicted | target))
    foreground_count = int(np.count_nonzero(predicted) + np.count_nonzero(target))

    return {
        "iou": intersection / union if union else 1.0,
        "dice": (2 * intersection / foreground_count) if foreground_count else 1.0,
    }
