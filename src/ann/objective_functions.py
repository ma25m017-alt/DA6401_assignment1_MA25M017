"""
Loss functions and gradients for the NumPy neural network implementation.
"""

from __future__ import annotations

import numpy as np

from ann.activations import softmax

EPSILON = 1e-12


def normalize_loss_name(name: str | None) -> str:
    if name is None:
        return 'cross_entropy'

    normalized = name.strip().lower().replace('-', '_')
    normalized = {
        'ce': 'cross_entropy',
        'crossentropy': 'cross_entropy',
        'mean_squared': 'mean_squared_error',
        'mse': 'mean_squared_error',
    }.get(normalized, normalized)

    if normalized not in {'cross_entropy', 'mean_squared_error'}:
        raise ValueError("Unsupported loss. Choose 'cross_entropy' or 'mean_squared_error'.")

    return normalized


def labels_from_targets(y_true: np.ndarray) -> np.ndarray:
    y_array = np.asarray(y_true)
    if y_array.ndim > 1 and y_array.shape[1] > 1:
        return np.argmax(y_array, axis=1).astype(np.int64)
    return y_array.reshape(-1).astype(np.int64)


def to_one_hot(y_true: np.ndarray, num_classes: int) -> np.ndarray:
    y_array = np.asarray(y_true)
    if y_array.ndim > 1 and y_array.shape[1] == num_classes:
        return y_array.astype(np.float64)

    labels = labels_from_targets(y_array)
    one_hot = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def cross_entropy_loss(logits: np.ndarray, y_true: np.ndarray) -> float:
    probabilities = softmax(logits)
    y_one_hot = to_one_hot(y_true, logits.shape[1])
    loss = -np.sum(y_one_hot * np.log(probabilities + EPSILON)) / logits.shape[0]
    return float(loss)


def cross_entropy_gradient(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    probabilities = softmax(logits)
    y_one_hot = to_one_hot(y_true, logits.shape[1])
    return (probabilities - y_one_hot) / logits.shape[0]


def mean_squared_error(logits: np.ndarray, y_true: np.ndarray) -> float:
    y_one_hot = to_one_hot(y_true, logits.shape[1])
    return float(np.mean((logits - y_one_hot) ** 2))


def mean_squared_error_gradient(logits: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    y_one_hot = to_one_hot(y_true, logits.shape[1])
    return 2.0 * (logits - y_one_hot) / logits.size


def get_loss_and_gradient(name: str | None):
    normalized = normalize_loss_name(name)
    if normalized == 'cross_entropy':
        return cross_entropy_loss, cross_entropy_gradient
    return mean_squared_error, mean_squared_error_gradient
