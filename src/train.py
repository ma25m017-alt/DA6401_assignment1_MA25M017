"""
Main training script.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from ann.activations import normalize_activation_name
from ann.neural_network import NeuralNetwork
from ann.objective_functions import normalize_loss_name
from ann.optimizers import normalize_optimizer_name
from utils.data_loader import load_dataset, normalize_dataset_name

SRC_DIR = Path(__file__).resolve().parent
BEST_CONFIG_PATH = SRC_DIR / 'best_config.json'

DEFAULT_BEST_CONFIG = {
    'dataset': 'mnist',
    'epochs': 10,
    'batch_size': 32,
    'loss': 'cross_entropy',
    'optimizer': 'nag',
    'learning_rate': 0.01,
    'weight_decay': 0.0001,
    'num_layers': 3,
    'hidden_size': [128, 128, 64],
    'activation': 'relu',
    'weight_init': 'xavier',
    'wandb_project': 'DA6401_A1_Mousumi',
    'model_path': 'best_model.npy',
}


def normalize_weight_init(name: str) -> str:
    normalized = str(name).strip().lower().replace('-', '_')
    if normalized not in {'random', 'xavier'}:
        raise argparse.ArgumentTypeError("weight_init must be 'random' or 'xavier'.")
    return normalized


def _normalize_hidden_size_defaults(hidden_size, num_layers):
    if hidden_size is None:
        values = list(DEFAULT_BEST_CONFIG['hidden_size'])
    elif isinstance(hidden_size, str):
        values = [int(part.strip()) for part in hidden_size.split(',') if part.strip()]
    else:
        values = []
        for item in hidden_size:
            if isinstance(item, str) and ',' in item:
                values.extend(int(part.strip()) for part in item.split(',') if part.strip())
            else:
                values.append(int(item))

    if num_layers == 0:
        return []

    if len(values) == 1 and num_layers > 1:
        values = values * num_layers

    if len(values) != num_layers:
        raise ValueError('hidden_size must have one value or exactly num_layers values.')

    return values


def load_best_config_defaults() -> dict:
    defaults = dict(DEFAULT_BEST_CONFIG)
    if not BEST_CONFIG_PATH.exists():
        return defaults

    try:
        with BEST_CONFIG_PATH.open('r', encoding='utf-8') as config_file:
            loaded = json.load(config_file)
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return defaults

    defaults.update(loaded)
    defaults['dataset'] = normalize_dataset_name(defaults.get('dataset', DEFAULT_BEST_CONFIG['dataset']))
    defaults['loss'] = normalize_loss_name(defaults.get('loss', DEFAULT_BEST_CONFIG['loss']))
    defaults['optimizer'] = normalize_optimizer_name(defaults.get('optimizer', DEFAULT_BEST_CONFIG['optimizer']))
    defaults['activation'] = normalize_activation_name(defaults.get('activation', DEFAULT_BEST_CONFIG['activation']))
    defaults['weight_init'] = normalize_weight_init(defaults.get('weight_init', DEFAULT_BEST_CONFIG['weight_init']))
    defaults['epochs'] = int(defaults.get('epochs', DEFAULT_BEST_CONFIG['epochs']))
    defaults['batch_size'] = int(defaults.get('batch_size', DEFAULT_BEST_CONFIG['batch_size']))
    defaults['learning_rate'] = float(defaults.get('learning_rate', DEFAULT_BEST_CONFIG['learning_rate']))
    defaults['weight_decay'] = float(defaults.get('weight_decay', DEFAULT_BEST_CONFIG['weight_decay']))
    defaults['num_layers'] = int(defaults.get('num_layers', DEFAULT_BEST_CONFIG['num_layers']))
    defaults['hidden_size'] = _normalize_hidden_size_defaults(
        defaults.get('hidden_size', DEFAULT_BEST_CONFIG['hidden_size']),
        defaults['num_layers'],
    )
    defaults['wandb_project'] = str(defaults.get('wandb_project', DEFAULT_BEST_CONFIG['wandb_project']))
    defaults['model_path'] = str(defaults.get('model_path', DEFAULT_BEST_CONFIG['model_path']))
    return defaults


BEST_CONFIG = load_best_config_defaults()


def _normalize_hidden_sizes(hidden_size, num_layers):
    if hidden_size is None:
        values = list(BEST_CONFIG['hidden_size'])
    elif isinstance(hidden_size, str):
        values = [int(part.strip()) for part in hidden_size.split(',') if part.strip()]
    else:
        values = []
        for item in hidden_size:
            if isinstance(item, str) and ',' in item:
                values.extend(int(part.strip()) for part in item.split(',') if part.strip())
            else:
                values.append(int(item))

    if num_layers == 0:
        return []

    if len(values) == 1 and num_layers > 1:
        values = values * num_layers

    if len(values) != num_layers:
        raise ValueError('hidden_size must have one value or exactly num_layers values.')

    return values


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument('-d', '--dataset', type=normalize_dataset_name, choices=['mnist', 'fashion'], default=BEST_CONFIG['dataset'])
    parser.add_argument('-e', '--epochs', type=int, default=BEST_CONFIG['epochs'])
    parser.add_argument('-b', '--batch_size', '--batch-size', type=int, default=BEST_CONFIG['batch_size'])
    parser.add_argument(
        '-l',
        '--loss',
        type=normalize_loss_name,
        choices=['cross_entropy', 'mean_squared_error'],
        default=BEST_CONFIG['loss'],
    )
    parser.add_argument(
        '-o',
        '--optimizer',
        type=normalize_optimizer_name,
        choices=['sgd', 'momentum', 'nag', 'rmsprop'],
        default=BEST_CONFIG['optimizer'],
    )
    parser.add_argument('-lr', '--learning_rate', '--learning-rate', type=float, default=BEST_CONFIG['learning_rate'])
    parser.add_argument('-wd', '--weight_decay', '--weight-decay', type=float, default=BEST_CONFIG['weight_decay'])
    parser.add_argument('-nhl', '--num_layers', '--num-layers', type=int, default=BEST_CONFIG['num_layers'])
    parser.add_argument(
        '-sz',
        '--hidden_size',
        '--hidden-size',
        nargs='+',
        default=[str(value) for value in BEST_CONFIG['hidden_size']],
    )
    parser.add_argument(
        '-a',
        '--activation',
        type=normalize_activation_name,
        choices=['sigmoid', 'tanh', 'relu'],
        default=BEST_CONFIG['activation'],
    )
    parser.add_argument(
        '-wi',
        '--weight_init',
        '--weight-init',
        type=normalize_weight_init,
        choices=['random', 'xavier'],
        default=BEST_CONFIG['weight_init'],
    )
    parser.add_argument('-wp', '--wandb_project', '--wandb-project', default=BEST_CONFIG['wandb_project'])
    parser.add_argument('-mp', '--model_path', '--model-path', default=BEST_CONFIG['model_path'])
    return parser


def parse_arguments(cli_args=None):
    parser = argparse.ArgumentParser(description='Train a neural network')
    add_common_arguments(parser)
    return parser.parse_args(args=cli_args)


def normalize_cli_config(args):
    args.hidden_size = _normalize_hidden_sizes(args.hidden_size, args.num_layers)
    return args


def config_from_args(args):
    return {
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'loss': args.loss,
        'optimizer': args.optimizer,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'num_layers': args.num_layers,
        'hidden_size': list(args.hidden_size),
        'activation': args.activation,
        'weight_init': args.weight_init,
        'wandb_project': args.wandb_project,
        'model_path': args.model_path,
    }


def resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if not path.is_absolute():
        path = SRC_DIR / path
    return path


def _init_wandb(args):
    if wandb is None or not args.wandb_project:
        return None

    return wandb.init(
        project=args.wandb_project,
        config=config_from_args(args),
        reinit=True,
        mode='offline',
    )


def main(cli_args=None):
    args = normalize_cli_config(parse_arguments(cli_args))

    (X_train, y_train), (X_test, y_test) = load_dataset(args.dataset)
    model = NeuralNetwork(args)
    run = _init_wandb(args)

    history, best_weights = model.train(
        X_train,
        y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        X_val=X_test,
        y_val=y_test,
    )

    model.set_weights(best_weights)
    metrics = model.evaluate(X_test, y_test, batch_size=args.batch_size)

    model_path = resolve_model_path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(model_path, best_weights)

    default_model_path = SRC_DIR / 'best_model.npy'
    if model_path.resolve() != default_model_path.resolve():
        np.save(default_model_path, best_weights)

    config_path = BEST_CONFIG_PATH
    with config_path.open('w', encoding='utf-8') as config_file:
        json.dump(config_from_args(args), config_file, indent=2)

    if run is not None:
        final_log = {
            'final_loss': metrics['loss'],
            'final_accuracy': metrics['accuracy'],
            'final_precision': metrics['precision'],
            'final_recall': metrics['recall'],
            'final_f1': metrics['f1'],
        }
        if history['train_loss']:
            final_log['train_loss'] = history['train_loss'][-1]
            final_log['train_accuracy'] = history['train_accuracy'][-1]
        if history['val_loss']:
            final_log['val_loss'] = history['val_loss'][-1]
        wandb.log(final_log)
        wandb.finish()

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1']:.4f}")
    print(f"Saved model to: {model_path}")
    print(f"Saved config to: {config_path}")
    return metrics


if __name__ == '__main__':
    main()
