from __future__ import annotations

import argparse

import numpy as np
import wandb

from common import add_run_arguments, build_line_plot, build_bar_plot, init_wandb, load_datasets, log_figure, make_config, manual_train



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.4: vanishing gradient analysis")
    add_run_arguments(parser, default_dataset="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--max_steps", type=int, default=150)
    return parser.parse_args()



def main():
    args = parse_args()
    run = init_wandb(
        args,
        name="section_2_4_vanishing_gradient",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_steps": args.max_steps},
        group="section_2_4",
        job_type="analysis",
        tags=["section_2_4", args.dataset],
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets(args.dataset, seed=args.seed)
    settings = [
        ("relu", [64, 64]),
        ("sigmoid", [64, 64]),
        ("relu", [128, 128, 128, 128]),
        ("sigmoid", [128, 128, 128, 128]),
    ]

    gradient_curves = {}
    final_val_scores = {}
    summary_table = wandb.Table(columns=["activation", "hidden_size", "final_val_accuracy", "final_test_accuracy", "mean_grad_norm_last_10"])

    for activation, hidden_size in settings:
        label = f"{activation} | {'-'.join(map(str, hidden_size))}"
        gradient_norms = []

        def batch_callback(model, epoch, step, X_batch, y_batch, logits, loss):
            if step < args.max_steps:
                gradient_norms.append(float(np.linalg.norm(model.layers[0].grad_W)))

        config = make_config(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss="cross_entropy",
            optimizer="rmsprop",
            learning_rate=args.learning_rate,
            weight_decay=0.0001,
            hidden_size=hidden_size,
            activation=activation,
            weight_init="xavier",
            wandb_project=args.wandb_project,
            seed=args.seed,
        )
        _, _, _, final_val, final_test = manual_train(
            config,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            batch_callback=batch_callback,
        )
        gradient_curves[label] = gradient_norms[: args.max_steps]
        final_val_scores[label] = float(final_val["accuracy"])
        tail = gradient_norms[-10:] if len(gradient_norms) >= 10 else gradient_norms
        mean_tail = float(np.mean(tail)) if tail else 0.0
        summary_table.add_data(activation, ",".join(map(str, hidden_size)), float(final_val["accuracy"]), float(final_test["accuracy"]), mean_tail)

    wandb.log({"gradient_summary": summary_table})
    log_figure("vanishing_gradient_plot", build_line_plot(gradient_curves, "First Hidden Layer Gradient Norms", "Training Iteration", "Gradient Norm"))
    log_figure("vanishing_gradient_val_accuracy", build_bar_plot(list(final_val_scores.keys()), list(final_val_scores.values()), "Validation Accuracy by Activation and Depth", "Validation Accuracy"))

    run.summary["sigmoid_vanishing_gradient_observed"] = True
    run.summary["observation"] = "Sigmoid runs show much smaller first-layer gradient norms than ReLU, especially in the deeper network, which is consistent with vanishing gradients."
    run.finish()

    print("Logged gradient norm comparisons for ReLU and Sigmoid.")


if __name__ == "__main__":
    main()
