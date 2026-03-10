"""
Dense neural layer implementation with cached activations and gradients.
"""

from __future__ import annotations

import numpy as np

from ann.activations import apply_activation, apply_activation_derivative, normalize_activation_name


class NeuralLayer:
    """
    Fully connected layer that stores gradients after each backward pass.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = 'relu',
        weight_init: str = 'xavier',
        seed: int | None = None,
    ) -> None:
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.activation = normalize_activation_name(activation)
        self.weight_init = weight_init.strip().lower()
        self.rng = np.random.default_rng(seed)

        self.W = self._initialize_weights()
        self.b = np.zeros(self.output_dim, dtype=np.float64)

        self.input_cache: np.ndarray | None = None
        self.linear_cache: np.ndarray | None = None
        self.output_cache: np.ndarray | None = None

        self.grad_W = np.zeros_like(self.W)
        self.grad_b = np.zeros_like(self.b)

    def _initialize_weights(self) -> np.ndarray:
        if self.weight_init == 'xavier':
            limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
            return self.rng.uniform(-limit, limit, size=(self.input_dim, self.output_dim)).astype(np.float64)

        if self.weight_init == 'random':
            return self.rng.normal(0.0, 0.01, size=(self.input_dim, self.output_dim)).astype(np.float64)

        raise ValueError("weight_init must be either 'random' or 'xavier'.")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        input_array = np.asarray(inputs, dtype=np.float64)
        if input_array.ndim == 1:
            input_array = input_array.reshape(1, -1)

        self.input_cache = input_array
        self.linear_cache = input_array @ self.W + self.b
        self.output_cache = apply_activation(self.activation, self.linear_cache)
        return self.output_cache

    def backward(self, grad_output: np.ndarray, weight_decay: float = 0.0) -> np.ndarray:
        if self.input_cache is None or self.linear_cache is None:
            raise RuntimeError('forward() must be called before backward().')

        grad_output_array = np.asarray(grad_output, dtype=np.float64)
        activation_grad = apply_activation_derivative(self.activation, self.linear_cache)
        grad_linear = grad_output_array * activation_grad

        self.grad_W = self.input_cache.T @ grad_linear + (weight_decay * self.W)
        self.grad_b = np.sum(grad_linear, axis=0)
        return grad_linear @ self.W.T
