from __future__ import annotations

import argparse

import wandb

from common import add_run_arguments, build_line_plot, fit_model, init_wandb, load_datasets, log_figure, make_config

OPTIMIZERS = ["sgd", "momentum", "nag", "rmsprop"]



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.3: optimizer showdown")
    add_run_arguments(parser, default_dataset="mnist")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    return parser.parse_args()



def main():
    args = parse_args()
    run = init_wandb(
        args,
        name="section_2_3_optimizer_showdown",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate},
        group="section_2_3",
        job_type="analysis",
        tags=["section_2_3", args.dataset],
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets(args.dataset, seed=args.seed)
    loss_curves = {}
    val_curves = {}
    summary_table = wandb.Table(columns=["optimizer", "final_train_loss", "final_val_accuracy", "final_test_accuracy", "final_test_f1"])

    best_optimizer = None
    best_loss = float("inf")
    for optimizer in OPTIMIZERS:
        config = make_config(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss="cross_entropy",
            optimizer=optimizer,
            learning_rate=args.learning_rate,
            weight_decay=0.0001,
            hidden_size=[128, 128, 128],
            activation="relu",
            weight_init="xavier",
            wandb_project=args.wandb_project,
            seed=args.seed,
        )
        _, history, _, train_metrics, val_metrics, test_metrics = fit_model(config, X_train, y_train, X_val, y_val, X_test, y_test)
        loss_curves[optimizer] = history["train_loss"]
        val_curves[optimizer] = history["val_accuracy"]
        summary_table.add_data(optimizer, float(train_metrics["loss"]), float(val_metrics["accuracy"]), float(test_metrics["accuracy"]), float(test_metrics["f1"]))
        if history["train_loss"][-1] < best_loss:
            best_loss = history["train_loss"][-1]
            best_optimizer = optimizer

    wandb.log({"optimizer_summary": summary_table})
    log_figure("optimizer_train_loss_plot", build_line_plot(loss_curves, "Optimizer Convergence in First 5 Epochs", "Epoch", "Training Loss"))
    log_figure("optimizer_val_accuracy_plot", build_line_plot(val_curves, "Validation Accuracy by Optimizer", "Epoch", "Validation Accuracy"))

    run.summary["fastest_optimizer_first_5_epochs"] = best_optimizer
    run.summary["rmsprop_theory"] = "RMSProp adapts per-parameter learning rates using recent squared gradients, which stabilizes updates and handles uneven curvature better than plain SGD."
    run.finish()

    print(f"Best optimizer over the first {args.epochs} epochs: {best_optimizer}")


if __name__ == "__main__":
    main()
