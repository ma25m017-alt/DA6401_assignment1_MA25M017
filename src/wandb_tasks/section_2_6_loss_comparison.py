from __future__ import annotations

import argparse

import wandb

from common import add_run_arguments, build_line_plot, fit_model, init_wandb, load_datasets, log_figure, make_config



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.6: loss function comparison")
    add_run_arguments(parser, default_dataset="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    return parser.parse_args()



def main():
    args = parse_args()
    run = init_wandb(
        args,
        name="section_2_6_loss_comparison",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.learning_rate},
        group="section_2_6",
        job_type="analysis",
        tags=["section_2_6", args.dataset],
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets(args.dataset, seed=args.seed)
    losses = ["cross_entropy", "mean_squared_error"]
    loss_curves = {}
    val_curves = {}
    summary_table = wandb.Table(columns=["loss", "final_train_loss", "final_val_accuracy", "final_test_accuracy", "final_test_f1"])

    fastest_loss = None
    best_train_loss = float("inf")
    for loss_name in losses:
        config = make_config(
            dataset=args.dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss=loss_name,
            optimizer="rmsprop",
            learning_rate=args.learning_rate,
            weight_decay=0.0001,
            hidden_size=[128, 64],
            activation="relu",
            weight_init="xavier",
            wandb_project=args.wandb_project,
            seed=args.seed,
        )
        _, history, _, train_metrics, val_metrics, test_metrics = fit_model(config, X_train, y_train, X_val, y_val, X_test, y_test)
        loss_curves[loss_name] = history["train_loss"]
        val_curves[loss_name] = history["val_accuracy"]
        summary_table.add_data(loss_name, float(train_metrics["loss"]), float(val_metrics["accuracy"]), float(test_metrics["accuracy"]), float(test_metrics["f1"]))
        if history["train_loss"][-1] < best_train_loss:
            best_train_loss = history["train_loss"][-1]
            fastest_loss = loss_name

    wandb.log({"loss_summary": summary_table})
    log_figure("loss_comparison_train_curve", build_line_plot(loss_curves, "Training Loss by Objective", "Epoch", "Training Loss"))
    log_figure("loss_comparison_val_accuracy", build_line_plot(val_curves, "Validation Accuracy by Objective", "Epoch", "Validation Accuracy"))

    run.summary["faster_converging_loss"] = fastest_loss
    run.summary["cross_entropy_theory"] = "Cross-entropy aligns directly with class probabilities and produces stronger gradients for wrong predictions than MSE in multi-class classification."
    run.finish()

    print(f"Faster converging loss: {fastest_loss}")


if __name__ == "__main__":
    main()
