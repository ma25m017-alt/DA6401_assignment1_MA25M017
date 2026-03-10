"""
Optimizer implementations used by the neural network.
"""

from __future__ import annotations

import numpy as np


def normalize_optimizer_name(name: str | None) -> str:
    if name is None:
        return 'sgd'

    normalized = name.strip().lower().replace('-', '_')
    normalized = {
        'gd': 'sgd',
        'gradient_descent': 'sgd',
        'nesterov': 'nag',
    }.get(normalized, normalized)

    if normalized not in {'sgd', 'momentum', 'nag', 'rmsprop'}:
        raise ValueError('Unsupported optimizer. Choose sgd, momentum, nag, or rmsprop.')

    return normalized


class BaseOptimizer:
    def __init__(self, layers, learning_rate: float = 0.01) -> None:
        self.layers = layers
        self.learning_rate = float(learning_rate)

    def step(self) -> None:
        raise NotImplementedError


class SGDOptimizer(BaseOptimizer):
    def step(self) -> None:
        for layer in self.layers:
            layer.W -= self.learning_rate * layer.grad_W
            layer.b -= self.learning_rate * layer.grad_b


class MomentumOptimizer(BaseOptimizer):
    def __init__(self, layers, learning_rate: float = 0.01, momentum: float = 0.9) -> None:
        super().__init__(layers, learning_rate)
        self.momentum = float(momentum)
        self.velocity_W = [np.zeros_like(layer.W) for layer in layers]
        self.velocity_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self) -> None:
        for index, layer in enumerate(self.layers):
            self.velocity_W[index] = (self.momentum * self.velocity_W[index]) - (
                self.learning_rate * layer.grad_W
            )
            self.velocity_b[index] = (self.momentum * self.velocity_b[index]) - (
                self.learning_rate * layer.grad_b
            )
            layer.W += self.velocity_W[index]
            layer.b += self.velocity_b[index]


class NAGOptimizer(MomentumOptimizer):
    def step(self) -> None:
        for index, layer in enumerate(self.layers):
            previous_velocity_W = self.velocity_W[index].copy()
            previous_velocity_b = self.velocity_b[index].copy()

            self.velocity_W[index] = (self.momentum * self.velocity_W[index]) - (
                self.learning_rate * layer.grad_W
            )
            self.velocity_b[index] = (self.momentum * self.velocity_b[index]) - (
                self.learning_rate * layer.grad_b
            )

            layer.W += (-self.momentum * previous_velocity_W) + (
                (1.0 + self.momentum) * self.velocity_W[index]
            )
            layer.b += (-self.momentum * previous_velocity_b) + (
                (1.0 + self.momentum) * self.velocity_b[index]
            )


class RMSPropOptimizer(BaseOptimizer):
    def __init__(
        self,
        layers,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(layers, learning_rate)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.cache_W = [np.zeros_like(layer.W) for layer in layers]
        self.cache_b = [np.zeros_like(layer.b) for layer in layers]

    def step(self) -> None:
        for index, layer in enumerate(self.layers):
            self.cache_W[index] = (self.beta * self.cache_W[index]) + (
                (1.0 - self.beta) * (layer.grad_W**2)
            )
            self.cache_b[index] = (self.beta * self.cache_b[index]) + (
                (1.0 - self.beta) * (layer.grad_b**2)
            )

            layer.W -= self.learning_rate * layer.grad_W / (np.sqrt(self.cache_W[index]) + self.epsilon)
            layer.b -= self.learning_rate * layer.grad_b / (np.sqrt(self.cache_b[index]) + self.epsilon)


def get_optimizer(name: str | None, layers, learning_rate: float = 0.01):
    normalized = normalize_optimizer_name(name)
    if normalized == 'sgd':
        return SGDOptimizer(layers, learning_rate=learning_rate)
    if normalized == 'momentum':
        return MomentumOptimizer(layers, learning_rate=learning_rate)
    if normalized == 'nag':
        return NAGOptimizer(layers, learning_rate=learning_rate)
    return RMSPropOptimizer(layers, learning_rate=learning_rate)
