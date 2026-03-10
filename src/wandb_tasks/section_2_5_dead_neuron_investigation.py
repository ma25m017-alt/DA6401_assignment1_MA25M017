from __future__ import annotations

import argparse

import numpy as np
import wandb

from common import activation_snapshot, add_run_arguments, build_histogram_plot, build_line_plot, init_wandb, load_datasets, log_figure, make_config, manual_train



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.5: dead neuron investigation")
    add_run_arguments(parser, default_dataset="mnist")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    return parser.parse_args()



def count_dead_neurons(model, probe_batch):
    activations = activation_snapshot(model, probe_batch)
    counts = []
    for layer_output in activations:
        dead_mask = np.all(layer_output == 0.0, axis=0)
        counts.append(int(dead_mask.sum()))
    return counts, activations[0]



def main():
    args = parse_args()
    run = init_wandb(
        args,
        name="section_2_5_dead_neuron_investigation",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate},
        group="section_2_5",
        job_type="analysis",
        tags=["section_2_5", args.dataset],
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets(args.dataset, seed=args.seed)
    probe_batch = X_val[:512]

    val_curves = {}
    dead_curves = {}
    histograms = {}
    gradient_curves = {}
    summary_table = wandb.Table(columns=["activation", "final_val_accuracy", "final_test_accuracy", "max_dead_neurons_layer1", "mean_first_layer_grad_norm"])

    for activation in ["relu", "tanh"]:
        dead_counts = []
        first_layer_grad_norms = []
        latest_histogram = None

        def batch_callback(model, epoch, step, X_batch, y_batch, logits, loss):
            first_layer_grad_norms.append(float(np.linalg.norm(model.layers[0].grad_W)))

        def epoch_callback(model, epoch, train_metrics, val_metrics, test_metrics):
            nonlocal latest_histogram
            layer_dead_counts, first_hidden = count_dead_neurons(model, probe_batch)
            dead_counts.append(layer_dead_counts[0])
            latest_histogram = first_hidden.copy()

        config = make_config(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss="cross_entropy",
            optimizer="sgd",
            learning_rate=args.learning_rate,
            weight_decay=0.0001,
            hidden_size=[128, 128, 128],
            activation=activation,
            weight_init="xavier",
            wandb_project=args.wandb_project,
            seed=args.seed,
        )
        _, history, _, final_val, final_test = manual_train(
            config,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            batch_callback=batch_callback,
            epoch_callback=epoch_callback,
        )
        val_curves[activation] = history["val_accuracy"]
        dead_curves[activation] = dead_counts
        histograms[activation] = latest_histogram if latest_histogram is not None else np.array([0.0])
        gradient_curves[activation] = first_layer_grad_norms[: max(1, min(200, len(first_layer_grad_norms)))]
        summary_table.add_data(
            activation,
            float(final_val["accuracy"]),
            float(final_test["accuracy"]),
            int(max(dead_counts) if dead_counts else 0),
            float(np.mean(first_layer_grad_norms) if first_layer_grad_norms else 0.0),
        )

    wandb.log({"dead_neuron_summary": summary_table})
    log_figure("dead_neuron_val_accuracy_plot", build_line_plot(val_curves, "Validation Accuracy: ReLU vs Tanh", "Epoch", "Validation Accuracy"))
    log_figure("dead_neuron_count_plot", build_line_plot(dead_curves, "Dead Neurons in First Hidden Layer", "Epoch", "Dead Neuron Count"))
    log_figure("dead_neuron_activation_histogram", build_histogram_plot(histograms, "Activation Distribution in First Hidden Layer", "Activation Value"))
    log_figure("dead_neuron_gradient_norm_plot", build_line_plot(gradient_curves, "Gradient Norms with High Learning Rate", "Iteration", "Gradient Norm"))

    run.summary["relu_plateau_explanation"] = "With a high learning rate, many ReLU units become permanently inactive and stop receiving gradient updates, which causes early accuracy plateaus."
    run.summary["tanh_comparison"] = "Tanh does not create dead neurons in the same way, so gradients continue to flow, although they can still shrink."
    run.finish()

    print("Logged ReLU dead-neuron and Tanh comparison plots.")


if __name__ == "__main__":
    main()
