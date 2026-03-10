"""
Inference script.
Evaluate trained models on test sets.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ann.neural_network import NeuralNetwork
from ann.objective_functions import labels_from_targets
from train import add_common_arguments, normalize_cli_config, resolve_model_path
from utils.data_loader import load_dataset


def parse_arguments(cli_args=None):
    """
    Parse command-line arguments for inference.
    """
    parser = argparse.ArgumentParser(description='Run inference on test set')
    add_common_arguments(parser)
    return parser.parse_args(args=cli_args)


def load_model(model_path):
    """
    Load trained model from disk.
    """
    path = resolve_model_path(str(model_path))
    if not path.exists():
        raise FileNotFoundError(f'Model file not found: {path}')
    return np.load(path, allow_pickle=True).item()


def evaluate_model(model, X_test, y_test, batch_size=32):
    """
    Evaluate model on test data.
    """
    metrics = model.evaluate(X_test, y_test, batch_size=batch_size)
    logits = metrics['logits']
    predictions = np.argmax(logits, axis=1)
    labels = labels_from_targets(y_test)

    accuracy = float(accuracy_score(labels, predictions))
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average='macro',
        zero_division=0,
    )

    return {
        'logits': logits,
        'loss': metrics['loss'],
        'accuracy': accuracy,
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
    }


def main(cli_args=None):
    """
    Main inference function.
    """
    args = normalize_cli_config(parse_arguments(cli_args))
    (_, _), (X_test, y_test) = load_dataset(args.dataset)

    model = NeuralNetwork(args)
    weights = load_model(args.model_path)
    model.set_weights(weights)

    metrics = evaluate_model(model, X_test, y_test, batch_size=args.batch_size)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    return metrics


if __name__ == '__main__':
    main()
