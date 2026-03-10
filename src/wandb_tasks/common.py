from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ann.neural_network import NeuralNetwork
from ann.objective_functions import labels_from_targets
from utils.data_loader import load_dataset, normalize_dataset_name

DEFAULT_PROJECT = "da6401_assignment_1_report"
DEFAULT_MODEL_PATH = SRC_DIR / "best_model.npy"
DEFAULT_CONFIG_PATH = SRC_DIR / "best_config.json"

CLASS_NAMES = {
    "mnist": [str(index) for index in range(10)],
    "fashion": [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ],
}


def add_run_arguments(parser: argparse.ArgumentParser, default_dataset: str = "mnist") -> argparse.ArgumentParser:
    parser.add_argument("--dataset", type=normalize_dataset_name, default=default_dataset)
    parser.add_argument("--wandb_project", default=DEFAULT_PROJECT)
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def class_names(dataset: str) -> list[str]:
    return list(CLASS_NAMES[normalize_dataset_name(dataset)])


def normalize_hidden_spec(hidden_spec) -> list[int]:
    if hidden_spec is None:
        return [128, 64]
    if isinstance(hidden_spec, str):
        return [int(part.strip()) for part in hidden_spec.split(",") if part.strip()]
    if np.isscalar(hidden_spec):
        return [int(hidden_spec)]

    values = []
    for item in hidden_spec:
        if isinstance(item, str) and "," in item:
            values.extend(int(part.strip()) for part in item.split(",") if part.strip())
        else:
            values.append(int(item))
    return values


def normalize_weight_init(name: str) -> str:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized not in {"random", "xavier"}:
        raise ValueError("weight_init must be either 'random' or 'xavier'.")
    return normalized


def make_config(
    dataset: str = "mnist",
    epochs: int = 10,
    batch_size: int = 64,
    loss: str = "cross_entropy",
    optimizer: str = "rmsprop",
    learning_rate: float = 0.001,
    weight_decay: float = 0.0001,
    hidden_size=None,
    activation: str = "relu",
    weight_init: str = "xavier",
    wandb_project: str = DEFAULT_PROJECT,
    seed: int = 42,
    model_path: str = "best_model.npy",
):
    hidden_sizes = normalize_hidden_spec(hidden_size)
    return SimpleNamespace(
        dataset=normalize_dataset_name(dataset),
        epochs=int(epochs),
        batch_size=int(batch_size),
        loss=str(loss),
        optimizer=str(optimizer),
        learning_rate=float(learning_rate),
        weight_decay=float(weight_decay),
        num_layers=len(hidden_sizes),
        hidden_size=hidden_sizes,
        activation=str(activation),
        weight_init=normalize_weight_init(weight_init),
        wandb_project=str(wandb_project),
        seed=int(seed),
        input_dim=784,
        output_dim=10,
        model_path=str(model_path),
    )


def config_to_dict(config) -> dict:
    return {
        "dataset": config.dataset,
        "epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "loss": config.loss,
        "optimizer": config.optimizer,
        "learning_rate": float(config.learning_rate),
        "weight_decay": float(config.weight_decay),
        "num_layers": int(config.num_layers),
        "hidden_size": list(config.hidden_size),
        "activation": config.activation,
        "weight_init": config.weight_init,
        "wandb_project": config.wandb_project,
        "seed": int(config.seed),
        "model_path": config.model_path,
    }


def load_json_config(path: Path | str) -> SimpleNamespace:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return make_config(**data)


def load_datasets(dataset: str, seed: int = 42, val_size: float = 0.1666667):
    (X_train_full, y_train_full), (X_test, y_test) = load_dataset(dataset)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size,
        random_state=seed,
        stratify=y_train_full,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def init_wandb(args, name: str, config: dict | None = None, group: str | None = None, job_type: str | None = None, tags=None):
    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=name,
        config=config,
        group=group,
        job_type=job_type,
        tags=list(tags or []),
        reinit="finish_previous",
    )


def log_figure(key: str, figure) -> None:
    wandb.log({key: wandb.Image(figure)})
    plt.close(figure)


def build_line_plot(curves: dict[str, list[float]], title: str, x_label: str, y_label: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, values in curves.items():
        if not values:
            continue
        x_values = np.arange(1, len(values) + 1)
        axis.plot(x_values, values, marker="o", linewidth=2, label=label)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    if curves:
        axis.legend()
    figure.tight_layout()
    return figure


def build_scatter_plot(x_values, y_values, labels, title: str, x_label: str, y_label: str, threshold: float | None = None) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(7, 6))
    x_array = np.asarray(x_values, dtype=float)
    y_array = np.asarray(y_values, dtype=float)
    axis.scatter(x_array, y_array, s=60, alpha=0.8)
    diagonal = [min(x_array.min(initial=0.0), y_array.min(initial=0.0)), max(x_array.max(initial=1.0), y_array.max(initial=1.0))]
    axis.plot(diagonal, diagonal, linestyle="--", color="black", linewidth=1)
    for x_val, y_val, label in zip(x_array, y_array, labels):
        axis.annotate(label, (x_val, y_val), fontsize=8, xytext=(5, 5), textcoords="offset points")
    if threshold is not None:
        axis.text(0.02, 0.98, f"highlight gap > {threshold:.2f}", transform=axis.transAxes, va="top")
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.3)
    figure.tight_layout()
    return figure


def build_bar_plot(labels, values, title: str, y_label: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(labels, values, color="#2c7fb8")
    axis.set_title(title)
    axis.set_ylabel(y_label)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    return figure


def build_histogram_plot(series_map: dict[str, np.ndarray], title: str, x_label: str) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(9, 5))
    for label, values in series_map.items():
        axis.hist(np.asarray(values).ravel(), bins=40, alpha=0.5, label=label)
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel("Frequency")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    return figure


def build_confusion_matrix_plot(y_true, y_pred, labels, title: str) -> plt.Figure:
    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(8, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
    display.plot(ax=axis, xticks_rotation=45, colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    return figure


def build_misclassified_grid(images, y_true, y_pred, labels, max_items: int = 20, title: str = "Misclassified Samples") -> plt.Figure:
    mistakes = np.where(np.asarray(y_true) != np.asarray(y_pred))[0][:max_items]
    if mistakes.size == 0:
        figure, axis = plt.subplots(figsize=(5, 3))
        axis.text(0.5, 0.5, "No misclassifications found", ha="center", va="center")
        axis.axis("off")
        figure.tight_layout()
        return figure

    cols = 4
    rows = int(np.ceil(len(mistakes) / cols))
    figure, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, index in zip(axes, mistakes):
        axis.imshow(images[index].reshape(28, 28), cmap="gray")
        axis.set_title(f"T:{labels[int(y_true[index])]} | P:{labels[int(y_pred[index])]}")
        axis.axis("off")
    for axis in axes[len(mistakes):]:
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    return figure


def activation_snapshot(model: NeuralNetwork, X_batch: np.ndarray) -> list[np.ndarray]:
    model.forward(X_batch)
    return [layer.output_cache.copy() for layer in model.layers[:-1]]


def zero_like_weights(model: NeuralNetwork) -> dict[str, np.ndarray]:
    return {key: np.zeros_like(value) for key, value in model.get_weights().items()}


def fit_model(config, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    model = NeuralNetwork(config)
    history, best_weights = model.train(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=config.batch_size,
        X_val=X_val,
        y_val=y_val,
    )
    model.set_weights(best_weights)
    val_metrics = model.evaluate(X_val, y_val, batch_size=config.batch_size)
    test_metrics = None
    if X_test is not None and y_test is not None:
        test_metrics = model.evaluate(X_test, y_test, batch_size=config.batch_size)
    train_metrics = {
        "loss": float(history["train_loss"][-1]),
        "accuracy": float(history["train_accuracy"][-1]),
    }
    return model, history, best_weights, train_metrics, val_metrics, test_metrics


def manual_train(
    config,
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    X_test=None,
    y_test=None,
    batch_callback: Callable | None = None,
    epoch_callback: Callable | None = None,
    weight_initializer: Callable[[NeuralNetwork], dict[str, np.ndarray]] | None = None,
):
    model = NeuralNetwork(config)
    if weight_initializer is not None:
        model.set_weights(weight_initializer(model))

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
        "test_accuracy": [],
        "test_f1": [],
    }
    best_score = -np.inf
    best_weights = model.get_weights()
    global_step = 0

    for epoch in range(int(config.epochs)):
        batch_losses = []
        batch_predictions = []
        batch_labels = []

        for X_batch, y_batch in model._iterate_minibatches(X_train, y_train, config.batch_size, shuffle=True):
            logits = model.forward(X_batch)
            loss = model.loss_function(logits, y_batch) + model._l2_penalty()
            model.backward(y_batch, logits)
            if batch_callback is not None:
                batch_callback(model, epoch, global_step, X_batch, y_batch, logits, float(loss))
            model.update_weights()

            batch_losses.append(float(loss))
            batch_predictions.append(np.argmax(logits, axis=1))
            batch_labels.append(labels_from_targets(y_batch))
            global_step += 1

        train_predictions = np.concatenate(batch_predictions)
        train_labels = np.concatenate(batch_labels)
        train_loss = float(np.mean(batch_losses))
        train_accuracy = float(np.mean(train_predictions == train_labels))
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)

        val_metrics = None
        test_metrics = None
        if X_val is not None and y_val is not None:
            val_metrics = model.evaluate(X_val, y_val, batch_size=config.batch_size)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_f1"].append(val_metrics["f1"])
        if X_test is not None and y_test is not None:
            test_metrics = model.evaluate(X_test, y_test, batch_size=config.batch_size)
            history["test_accuracy"].append(test_metrics["accuracy"])
            history["test_f1"].append(test_metrics["f1"])

        score = val_metrics["f1"] if val_metrics is not None else train_accuracy
        if score > best_score:
            best_score = score
            best_weights = model.get_weights()

        if epoch_callback is not None:
            epoch_callback(
                model,
                epoch,
                {"loss": train_loss, "accuracy": train_accuracy},
                val_metrics,
                test_metrics,
            )

    model.set_weights(best_weights)
    final_val = model.evaluate(X_val, y_val, batch_size=config.batch_size) if X_val is not None and y_val is not None else None
    final_test = model.evaluate(X_test, y_test, batch_size=config.batch_size) if X_test is not None and y_test is not None else None
    return model, history, best_weights, final_val, final_test


def save_artifacts(weights: dict[str, np.ndarray], config, model_path: Path | str = DEFAULT_MODEL_PATH, config_path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    model_file = Path(model_path)
    config_file = Path(config_path)
    np.save(model_file, weights)
    with config_file.open("w", encoding="utf-8") as handle:
        json.dump(config_to_dict(config), handle, indent=2)


def api_project_path(entity: str | None, project: str) -> str:
    if entity:
        return f"{entity}/{project}"
    return project
