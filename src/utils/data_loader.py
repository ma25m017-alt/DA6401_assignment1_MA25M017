"""
Dataset loading and preprocessing utilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def normalize_dataset_name(name: str) -> str:
    normalized = name.strip().lower().replace('-', '_')
    normalized = {
        'fashion_mnist': 'fashion',
        'fashion': 'fashion',
    }.get(normalized, normalized)

    if normalized not in {'mnist', 'fashion'}:
        raise ValueError("dataset must be either 'mnist' or 'fashion'.")

    return normalized


def _flatten_and_normalize(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1).astype(np.float64) / 255.0


def _load_from_local_keras_cache(dataset_name: str):
    cache_map = {
        'mnist': Path.home() / '.keras' / 'datasets' / 'mnist.npz',
        'fashion': Path.home() / '.keras' / 'datasets' / 'fashion-mnist.npz',
    }
    cache_path = cache_map[dataset_name]
    if not cache_path.exists():
        return None

    with np.load(cache_path, allow_pickle=False) as data:
        X_train = data['x_train']
        y_train = data['y_train']
        X_test = data['x_test']
        y_test = data['y_test']
    return (X_train, y_train), (X_test, y_test)


def _load_from_tensorflow(dataset_name: str):
    try:
        from tensorflow.keras.datasets import fashion_mnist, mnist
    except Exception:
        return None

    if dataset_name == 'mnist':
        return mnist.load_data()
    return fashion_mnist.load_data()


def _load_from_keras(dataset_name: str):
    try:
        from keras.datasets import fashion_mnist, mnist
    except Exception:
        return None

    try:
        if dataset_name == 'mnist':
            return mnist.load_data()
        return fashion_mnist.load_data()
    except Exception:
        return None


def _load_from_openml(dataset_name: str):
    from sklearn.datasets import fetch_openml

    openml_name = 'mnist_784' if dataset_name == 'mnist' else 'Fashion-MNIST'
    dataset = fetch_openml(name=openml_name, version=1, as_frame=False, parser='auto')
    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.int64)

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return (X_train, y_train), (X_test, y_test)


def load_dataset(dataset_name: str):
    normalized = normalize_dataset_name(dataset_name)

    loaders = (
        _load_from_local_keras_cache,
        _load_from_tensorflow,
        _load_from_keras,
        _load_from_openml,
    )

    data = None
    last_error = None
    for loader in loaders:
        try:
            data = loader(normalized)
        except Exception as exc:  # pragma: no cover
            last_error = exc
            data = None
        if data is not None:
            break

    if data is None:
        raise RuntimeError(
            'Unable to load dataset from local cache, TensorFlow/Keras, or OpenML.'
        ) from last_error

    (X_train, y_train), (X_test, y_test) = data
    return (
        _flatten_and_normalize(np.asarray(X_train)),
        np.asarray(y_train, dtype=np.int64),
    ), (
        _flatten_and_normalize(np.asarray(X_test)),
        np.asarray(y_test, dtype=np.int64),
    )
