from typing import Dict, Tuple, Union

import numpy as np
import torch


def ensure_numpy(data):
    """
    Converts input to a numpy array if it is a torch tensor.
    Handles scalars, lists, and arrays safely.
    """
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    return np.array(data)


def calculate_metrics(
    preds: Union[torch.Tensor, np.ndarray],
    probs: Union[torch.Tensor, np.ndarray],
    labels: Union[torch.Tensor, np.ndarray],
    threshold: float = 0.5
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """
    Calculates metrics using NumPy logic, accepting either Tensors or Arrays.
    """
    preds = ensure_numpy(preds)
    probs = ensure_numpy(probs)
    labels = ensure_numpy(labels)

    tp = np.sum((preds == 1) & (labels == 1))
    tn = np.sum((preds == 0) & (labels == 0))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    acc  = (tp + tn) / len(labels)
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1   = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0

    desc_score_indices = np.argsort(probs)[::-1]
    probs_sorted = probs[desc_score_indices]
    labels_sorted = labels[desc_score_indices]

    tps = np.cumsum(labels_sorted)
    fps = np.cumsum(1 - labels_sorted)

    distinct_indices = np.where(np.diff(probs_sorted))[0]
    threshold_idxs = np.r_[distinct_indices, len(labels_sorted) - 1]

    tpr = tps[threshold_idxs] / tps[-1] if tps[-1] > 0 else np.zeros_like(tps[threshold_idxs])
    fpr = fps[threshold_idxs] / fps[-1] if fps[-1] > 0 else np.zeros_like(fps[threshold_idxs])

    tpr = np.r_[0, tpr]
    fpr = np.r_[0, fpr]
    auc = np.trapz(tpr, fpr)

    return {
        'acc': float(acc),
        'auc': float(auc),
        'f1': float(f1),
        'sens': float(sens),
        'spec': float(spec),
    }, {
        'roc': (fpr, tpr, probs_sorted[threshold_idxs]),
        'cm': np.array([[tn, fp], [fn, tp]])
    }


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """
    Aggregates per-fold scalar metrics into mean and standard deviation.
    """
    keys = ['acc', 'auc', 'f1', 'sens', 'spec']
    aggregated = {}
    for key in keys:
        values = [fold[key] for fold in fold_metrics if key in fold]
        aggregated[key] = float(np.mean(values))
        aggregated[f'std_{key}'] = float(np.std(values))
    return aggregated
