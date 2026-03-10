"""
Activation functions used by the NumPy neural network implementation.
"""

from __future__ import annotations

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return sig * (1.0 - sig)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    tanh_x = np.tanh(x)
    return 1.0 - tanh_x**2


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=np.float64)


def softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x, axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / np.sum(exp_shifted, axis=1, keepdims=True)


ACTIVATION_FUNCTIONS = {
    'relu': (relu, relu_derivative),
    'sigmoid': (sigmoid, sigmoid_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative),
}


def normalize_activation_name(name: str | None) -> str:
    if name is None:
        return 'relu'

    normalized = name.strip().lower().replace('-', '_')
    normalized = {'identity': 'linear'}.get(normalized, normalized)

    if normalized not in ACTIVATION_FUNCTIONS:
        valid = ', '.join(sorted(ACTIVATION_FUNCTIONS))
        raise ValueError(f"Unsupported activation '{name}'. Choose from: {valid}.")

    return normalized


def get_activation(name: str | None):
    normalized = normalize_activation_name(name)
    return ACTIVATION_FUNCTIONS[normalized]


def apply_activation(name: str | None, x: np.ndarray) -> np.ndarray:
    activation, _ = get_activation(name)
    return activation(x)


def apply_activation_derivative(name: str | None, x: np.ndarray) -> np.ndarray:
    _, derivative = get_activation(name)
    return derivative(x)
