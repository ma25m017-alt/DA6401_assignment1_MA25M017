from __future__ import annotations

import argparse

import wandb

from common import add_run_arguments, build_bar_plot, fit_model, init_wandb, load_datasets, log_figure, make_config

CANDIDATE_CONFIGS = [
    {
        "name": "mnist_winner_style",
        "hidden_size": [128, 128, 64],
        "optimizer": "rmsprop",
        "activation": "relu",
        "learning_rate": 0.001,
    },
    {
        "name": "relu_nag_deeper",
        "hidden_size": [128, 128, 128],
        "optimizer": "nag",
        "activation": "relu",
        "learning_rate": 0.01,
    },
    {
        "name": "tanh_rmsprop_balanced",
        "hidden_size": [128, 64],
        "optimizer": "rmsprop",
        "activation": "tanh",
        "learning_rate": 0.001,
    },
]



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.10: Fashion-MNIST transfer challenge")
    add_run_arguments(parser, default_dataset="fashion")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=64)
    return parser.parse_args()



def main():
    args = parse_args()
    run = init_wandb(
        args,
        name="section_2_10_fashion_transfer",
        config={"epochs": args.epochs, "batch_size": args.batch_size},
        group="section_2_10",
        job_type="analysis",
        tags=["section_2_10", "fashion"],
    )

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets("fashion", seed=args.seed)
    summary_table = wandb.Table(columns=["name", "hidden_size", "optimizer", "activation", "test_accuracy", "test_f1"])
    accuracies = {}
    best_name = None
    best_accuracy = -1.0

    for candidate in CANDIDATE_CONFIGS:
        config = make_config(
            dataset="fashion",
            epochs=args.epochs,
            batch_size=args.batch_size,
            loss="cross_entropy",
            optimizer=candidate["optimizer"],
            learning_rate=candidate["learning_rate"],
            weight_decay=0.0001,
            hidden_size=candidate["hidden_size"],
            activation=candidate["activation"],
            weight_init="xavier",
            wandb_project=args.wandb_project,
            seed=args.seed,
        )
        _, _, _, _, _, test_metrics = fit_model(config, X_train, y_train, X_val, y_val, X_test, y_test)
        accuracies[candidate["name"]] = float(test_metrics["accuracy"])
        summary_table.add_data(
            candidate["name"],
            ",".join(map(str, candidate["hidden_size"])),
            candidate["optimizer"],
            candidate["activation"],
            float(test_metrics["accuracy"]),
            float(test_metrics["f1"]),
        )
        if test_metrics["accuracy"] > best_accuracy:
            best_accuracy = float(test_metrics["accuracy"])
            best_name = candidate["name"]

    wandb.log({"fashion_transfer_summary": summary_table})
    log_figure("fashion_transfer_accuracy_plot", build_bar_plot(list(accuracies.keys()), list(accuracies.values()), "Fashion-MNIST Accuracy for 3 Selected Configurations", "Test Accuracy"))

    run.summary["best_fashion_configuration"] = best_name
    run.summary["did_mnist_best_transfer"] = best_name == "mnist_winner_style"
    run.summary["transfer_note"] = "Fashion-MNIST has greater intra-class variation and more visually similar categories than MNIST, so the best digit configuration may not always remain best for clothing."
    run.finish()

    print(f"Best Fashion-MNIST configuration: {best_name}")


if __name__ == "__main__":
    main()
