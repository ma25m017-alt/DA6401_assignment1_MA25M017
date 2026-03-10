from __future__ import annotations

import argparse
import numpy as np
import wandb

from utils.data_loader import normalize_dataset_name
from common import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MODEL_PATH,
    build_confusion_matrix_plot,
    build_misclassified_grid,
    class_names,
    init_wandb,
    load_datasets,
    load_json_config,
    log_figure,
)
from ann.activations import softmax
from ann.neural_network import NeuralNetwork



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.8: error analysis for the best model")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--wandb_project", default="da6401_assignment_1_report")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    return parser.parse_args()



def main():
    args = parse_args()
    model_config = load_json_config(args.config_path)
    if args.dataset:
        model_config.dataset = normalize_dataset_name(args.dataset)

    run = init_wandb(
        args,
        name="section_2_8_error_analysis",
        config={"model_path": args.model_path, "config_path": args.config_path, "dataset": model_config.dataset},
        group="section_2_8",
        job_type="analysis",
        tags=["section_2_8", model_config.dataset],
    )

    _, _, (X_test, y_test) = load_datasets(model_config.dataset, seed=args.seed)
    weights = np.load(args.model_path, allow_pickle=True).item()
    model = NeuralNetwork(model_config)
    model.set_weights(weights)

    logits = model.forward(X_test)
    probabilities = softmax(logits)
    predictions = np.argmax(logits, axis=1)
    labels = class_names(model_config.dataset)

    summary_table = wandb.Table(columns=["metric", "value"])
    metrics = model.evaluate(X_test, y_test, batch_size=model_config.batch_size)
    for key in ["accuracy", "precision", "recall", "f1", "loss"]:
        summary_table.add_data(key, float(metrics[key]))
    wandb.log({"error_analysis_metrics": summary_table})

    failure_table = wandb.Table(columns=["image", "true_label", "predicted_label", "confidence"])
    wrong_indices = np.where(predictions != y_test)[0]
    ranked_wrong = wrong_indices[np.argsort(probabilities[wrong_indices].max(axis=1))[::-1][:25]]
    for index in ranked_wrong:
        failure_table.add_data(
            wandb.Image(X_test[index].reshape(28, 28)),
            labels[int(y_test[index])],
            labels[int(predictions[index])],
            float(probabilities[index].max()),
        )
    wandb.log({"hard_failure_gallery": failure_table})

    log_figure("confusion_matrix", build_confusion_matrix_plot(y_test, predictions, labels, "Confusion Matrix for Best Model"))
    log_figure("misclassified_samples", build_misclassified_grid(X_test, y_test, predictions, labels, max_items=20, title="Creative Failure Gallery"))

    run.summary["best_model_accuracy"] = metrics["accuracy"]
    run.summary["best_model_f1"] = metrics["f1"]
    run.finish()

    print("Logged confusion matrix and misclassification gallery.")


if __name__ == "__main__":
    main()
