"""
Main Neural Network Model class
Handles forward and backward propagation loops
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.activations import softmax
from ann.neural_layer import NeuralLayer
from ann.objective_functions import get_loss_and_gradient, labels_from_targets
from ann.optimizers import get_optimizer


class NeuralNetwork:
    """
    Main model class that orchestrates the neural network training and inference.
    """

    def __init__(self, cli_args):
        self.cli_args = cli_args
        self.input_dim = int(getattr(cli_args, 'input_dim', 784))
        self.output_dim = int(getattr(cli_args, 'output_dim', 10))
        self.learning_rate = float(getattr(cli_args, 'learning_rate', 0.001))
        self.weight_decay = float(getattr(cli_args, 'weight_decay', 0.0))
        self.activation = getattr(cli_args, 'activation', 'relu')
        self.weight_init = getattr(cli_args, 'weight_init', 'xavier')
        self.loss_name = getattr(cli_args, 'loss', 'cross_entropy')
        self.optimizer_name = getattr(cli_args, 'optimizer', 'sgd')

        hidden_size_arg = getattr(cli_args, 'hidden_size', [128, 64])
        provided_num_layers = getattr(cli_args, 'num_layers', None)
        if provided_num_layers is None:
            if isinstance(hidden_size_arg, (list, tuple, np.ndarray)):
                num_layers = len(hidden_size_arg)
            else:
                num_layers = 1
        else:
            num_layers = int(provided_num_layers)
        self.hidden_sizes = self._normalize_hidden_sizes(hidden_size_arg, num_layers)

        self.layers = []
        previous_dim = self.input_dim
        seed = int(getattr(cli_args, 'seed', 42))
        for index, hidden_dim in enumerate(self.hidden_sizes):
            self.layers.append(
                NeuralLayer(
                    previous_dim,
                    hidden_dim,
                    activation=self.activation,
                    weight_init=self.weight_init,
                    seed=seed + index,
                )
            )
            previous_dim = hidden_dim

        self.layers.append(
            NeuralLayer(
                previous_dim,
                self.output_dim,
                activation='linear',
                weight_init=self.weight_init,
                seed=seed + len(self.hidden_sizes),
            )
        )

        self.loss_function, self.loss_gradient = get_loss_and_gradient(self.loss_name)
        self.optimizer = get_optimizer(self.optimizer_name, self.layers, learning_rate=self.learning_rate)

        self.grad_W = np.empty(len(self.layers), dtype=object)
        self.grad_b = np.empty(len(self.layers), dtype=object)
        self.latest_logits = None
        self.latest_probabilities = None

    @staticmethod
    def _normalize_hidden_sizes(hidden_size, num_layers):
        if hidden_size is None:
            hidden_sizes = [128] * max(num_layers, 1)
        elif isinstance(hidden_size, str):
            hidden_sizes = [int(part.strip()) for part in hidden_size.split(',') if part.strip()]
        elif np.isscalar(hidden_size):
            hidden_sizes = [int(hidden_size)]
        else:
            hidden_sizes = []
            for value in hidden_size:
                if isinstance(value, str) and ',' in value:
                    hidden_sizes.extend(int(part.strip()) for part in value.split(',') if part.strip())
                else:
                    hidden_sizes.append(int(value))

        if num_layers == 0:
            return []

        if len(hidden_sizes) == 1 and num_layers > 1:
            hidden_sizes = hidden_sizes * num_layers

        if len(hidden_sizes) != num_layers:
            raise ValueError('hidden_size must either contain one value or exactly num_layers values.')

        return hidden_sizes

    def forward(self, X):
        """
        Forward propagation through all layers.
        Returns logits only (no softmax applied to the returned output).
        X is shape (b, D_in) and output is shape (b, D_out).
        b is batch size, D_in is input dimension, D_out is output dimension.
        """
        activations = np.asarray(X, dtype=np.float64)
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)

        for layer in self.layers:
            activations = layer.forward(activations)

        logits = activations
        self.latest_logits = logits
        self.latest_probabilities = None
        return logits

    def predict_proba(self, X):
        """
        Convenience helper for callers that explicitly need softmax probabilities.
        """
        logits = self.forward(X)
        probabilities = softmax(logits)
        self.latest_probabilities = probabilities
        return probabilities

    def backward(self, y_true, y_pred):
        """
        Backward propagation to compute gradients.
        Returns two numpy arrays: grad_Ws, grad_bs.
        - `grad_Ws[0]` is gradient for the last (output) layer weights,
          `grad_bs[0]` is gradient for the last layer biases, and so on.
        """
        logits = np.asarray(y_pred, dtype=np.float64)
        gradient = self.loss_gradient(logits, y_true)

        grad_W_list = []
        grad_b_list = []
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, weight_decay=self.weight_decay)
            grad_W_list.append(layer.grad_W.copy())
            grad_b_list.append(layer.grad_b.copy())

        self.grad_W = np.empty(len(grad_W_list), dtype=object)
        self.grad_b = np.empty(len(grad_b_list), dtype=object)
        for index, (grad_w, grad_b) in enumerate(zip(grad_W_list, grad_b_list)):
            self.grad_W[index] = grad_w
            self.grad_b[index] = grad_b
        return self.grad_W, self.grad_b

    def update_weights(self):
        self.optimizer.step()

    def _iterate_minibatches(self, X, y, batch_size, shuffle=True):
        features = np.asarray(X, dtype=np.float64)
        targets = np.asarray(y)
        indices = np.arange(features.shape[0])
        if shuffle:
            np.random.shuffle(indices)

        for start in range(0, features.shape[0], batch_size):
            batch_indices = indices[start : start + batch_size]
            yield features[batch_indices], targets[batch_indices]

    def _l2_penalty(self):
        if self.weight_decay <= 0.0:
            return 0.0
        return 0.5 * self.weight_decay * sum(np.sum(layer.W**2) for layer in self.layers)

    def train(self, X_train, y_train, epochs=1, batch_size=32, X_val=None, y_val=None):
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
        }

        best_f1 = -np.inf
        best_weights = self.get_weights()

        for _ in range(int(epochs)):
            batch_losses = []
            predictions = []
            labels = []

            for X_batch, y_batch in self._iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                logits = self.forward(X_batch)
                loss = self.loss_function(logits, y_batch) + self._l2_penalty()
                self.backward(y_batch, logits)
                self.update_weights()

                batch_losses.append(loss)
                predictions.append(np.argmax(logits, axis=1))
                labels.append(labels_from_targets(y_batch))

            train_predictions = np.concatenate(predictions)
            train_labels = np.concatenate(labels)
            history['train_loss'].append(float(np.mean(batch_losses)))
            history['train_accuracy'].append(float(accuracy_score(train_labels, train_predictions)))

            if X_val is not None and y_val is not None:
                metrics = self.evaluate(X_val, y_val, batch_size=batch_size)
                history['val_loss'].append(metrics['loss'])
                history['val_accuracy'].append(metrics['accuracy'])
                history['val_precision'].append(metrics['precision'])
                history['val_recall'].append(metrics['recall'])
                history['val_f1'].append(metrics['f1'])

                if metrics['f1'] > best_f1:
                    best_f1 = metrics['f1']
                    best_weights = self.get_weights()
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(
                    train_labels,
                    train_predictions,
                    average='macro',
                    zero_division=0,
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_weights = self.get_weights()

        return history, best_weights

    def evaluate(self, X, y, batch_size=32):
        logits_batches = []
        labels = []
        batch_losses = []

        for X_batch, y_batch in self._iterate_minibatches(X, y, batch_size, shuffle=False):
            logits = self.forward(X_batch)
            logits_batches.append(logits)
            labels.append(labels_from_targets(y_batch))
            batch_losses.append(self.loss_function(logits, y_batch) + self._l2_penalty())

        logits = np.vstack(logits_batches)
        y_true = np.concatenate(labels)
        y_pred = np.argmax(logits, axis=1)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average='macro',
            zero_division=0,
        )

        return {
            'logits': logits,
            'loss': float(np.mean(batch_losses)),
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
        }

    def get_weights(self):
        d = {}
        for i, layer in enumerate(self.layers):
            d[f'W{i}'] = layer.W.copy()
            d[f'b{i}'] = layer.b.copy()
        return d

    def set_weights(self, weight_dict):
        for i, layer in enumerate(self.layers):
            w_key = f'W{i}'
            b_key = f'b{i}'
            if w_key in weight_dict:
                layer.W = weight_dict[w_key].copy()
            if b_key in weight_dict:
                layer.b = weight_dict[b_key].copy()
