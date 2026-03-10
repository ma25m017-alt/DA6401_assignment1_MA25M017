from __future__ import annotations

import argparse
from collections import defaultdict

import wandb

from common import (
    add_run_arguments,
    api_project_path,
    build_bar_plot,
    init_wandb,
    load_datasets,
    log_figure,
    make_config,
    normalize_hidden_spec,
)
from ann.neural_network import NeuralNetwork


SWEEP_PARAMETERS = {
    "optimizer": {"values": ["sgd", "momentum", "nag", "rmsprop"]},
    "learning_rate": {"values": [0.1, 0.01, 0.001]},
    "batch_size": {"values": [32, 64, 128]},
    "activation": {"values": ["relu", "tanh", "sigmoid"]},
    "hidden_spec": {"values": ["64,64", "128,64", "128,128,64"]},
    "loss": {"values": ["cross_entropy", "mean_squared_error"]},
    "weight_decay": {"values": [0.0, 0.0001]},
    "weight_init": {"values": ["random", "xavier"]},
}


PARAMETER_ORDER = [
    "optimizer",
    "learning_rate",
    "batch_size",
    "activation",
    "hidden_spec",
    "loss",
    "weight_decay",
    "weight_init",
]



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.2: hyperparameter sweep")
    add_run_arguments(parser, default_dataset="mnist")
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--sweep_id", default=None)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--log_test_metrics", action="store_true")
    return parser.parse_args()



def _value_key(value):
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)



def _compute_hyperparameter_impacts(runs):
    impacts = {}
    for parameter in PARAMETER_ORDER:
        grouped_scores = defaultdict(list)
        for run in runs:
            summary = run.summary._json_dict
            val_accuracy = summary.get("val_accuracy")
            config_value = run.config.get(parameter)
            if val_accuracy is None or config_value is None:
                continue
            grouped_scores[_value_key(config_value)].append(float(val_accuracy))

        if len(grouped_scores) < 2:
            continue

        mean_scores = {
            key: sum(values) / len(values)
            for key, values in grouped_scores.items()
        }
        impacts[parameter] = {
            "score": max(mean_scores.values()) - min(mean_scores.values()),
            "means": mean_scores,
        }
    return impacts



def _best_run_summary(run):
    summary = run.summary._json_dict
    best = {
        "name": run.name,
        "id": run.id,
        "val_accuracy": float(summary["val_accuracy"]),
        "val_f1": float(summary.get("val_f1", 0.0)),
        "train_accuracy": float(summary.get("train_accuracy", 0.0)),
        "configuration": {
            "optimizer": run.config.get("optimizer"),
            "learning_rate": run.config.get("learning_rate"),
            "batch_size": run.config.get("batch_size"),
            "activation": run.config.get("activation"),
            "hidden_spec": run.config.get("hidden_spec"),
            "loss": run.config.get("loss"),
            "weight_decay": run.config.get("weight_decay"),
            "weight_init": run.config.get("weight_init"),
        },
    }
    if summary.get("test_accuracy") is not None:
        best["test_accuracy"] = float(summary["test_accuracy"])
    if summary.get("test_f1") is not None:
        best["test_f1"] = float(summary["test_f1"])
    return best



def analyze_sweep(args, sweep_id):
    api = wandb.Api()
    project_path = api_project_path(args.wandb_entity, args.wandb_project)
    sweep = api.sweep(f"{project_path}/{sweep_id}")
    completed_runs = [
        run
        for run in sweep.runs
        if run.state == "finished" and run.summary.get("val_accuracy") is not None
    ]

    if not completed_runs:
        raise RuntimeError("No completed sweep runs with validation accuracy were found for analysis.")

    best_run = max(
        completed_runs,
        key=lambda candidate: candidate.summary.get("val_accuracy", float("-inf")),
    )
    best_summary = _best_run_summary(best_run)
    impacts = _compute_hyperparameter_impacts(completed_runs)
    most_significant = None
    if impacts:
        most_significant = max(impacts.items(), key=lambda item: item[1]["score"])[0]

    analysis_run = init_wandb(
        args,
        name=f"section_2_2_summary_{sweep_id}",
        config={
            "dataset": args.dataset,
            "sweep_id": sweep_id,
            "count": args.count,
            "epochs": args.epochs,
            "log_test_metrics": args.log_test_metrics,
        },
        group="section_2_2",
        job_type="analysis",
        tags=["section_2_2", args.dataset, "summary"],
    )

    results_table = wandb.Table(
        columns=[
            "run_name",
            "optimizer",
            "learning_rate",
            "batch_size",
            "activation",
            "hidden_spec",
            "loss",
            "weight_decay",
            "weight_init",
            "val_accuracy",
            "val_f1",
            "test_accuracy",
            "test_f1",
        ]
    )
    for run in completed_runs:
        summary = run.summary._json_dict
        results_table.add_data(
            run.name,
            run.config.get("optimizer"),
            run.config.get("learning_rate"),
            run.config.get("batch_size"),
            run.config.get("activation"),
            run.config.get("hidden_spec"),
            run.config.get("loss"),
            run.config.get("weight_decay"),
            run.config.get("weight_init"),
            float(summary.get("val_accuracy", 0.0)),
            float(summary.get("val_f1", 0.0)),
            summary.get("test_accuracy"),
            summary.get("test_f1"),
        )
    wandb.log({"sweep_results": results_table})

    if impacts:
        log_figure(
            "hyperparameter_impact_plot",
            build_bar_plot(
                list(impacts.keys()),
                [details["score"] for details in impacts.values()],
                "Hyperparameter Impact on Validation Accuracy",
                "Validation Accuracy Spread",
            ),
        )
        impact_table = wandb.Table(
            columns=["hyperparameter", "impact_score", "group_mean_summary"]
        )
        for parameter, details in impacts.items():
            mean_summary = ", ".join(
                f"{key}: {value:.4f}"
                for key, value in sorted(details["means"].items(), key=lambda item: item[0])
            )
            impact_table.add_data(parameter, float(details["score"]), mean_summary)
        wandb.log({"hyperparameter_impact_table": impact_table})

    analysis_run.summary["sweep_id"] = sweep_id
    analysis_run.summary["completed_runs"] = len(completed_runs)
    analysis_run.summary["most_significant_hyperparameter"] = most_significant
    analysis_run.summary["best_run"] = best_summary
    analysis_run.summary["selection_metric"] = "val_accuracy"
    analysis_run.summary["analysis_note"] = (
        "Best configuration is selected using validation accuracy only. "
        "Use the W&B Parallel Coordinates plot alongside this summary in the report."
    )
    analysis_run.finish()

    print(f"Most significant hyperparameter: {most_significant}")
    print(f"Best configuration: {best_summary['configuration']}")



def main():
    args = parse_args()
    if args.wandb_mode != "online":
        print("Warning: W&B sweeps and post-sweep analysis work best in online mode.")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_datasets(args.dataset, seed=args.seed)

    sweep_config = {
        "method": "random",
        "metric": {"name": "val_accuracy", "goal": "maximize"},
        "parameters": SWEEP_PARAMETERS,
    }

    def train_sweep_run():
        with wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            mode=args.wandb_mode,
            group="section_2_2",
            job_type="sweep_run",
            tags=["section_2_2", args.dataset],
            reinit="finish_previous",
        ) as run:
            hidden_sizes = normalize_hidden_spec(run.config.hidden_spec)
            config = make_config(
                dataset=args.dataset,
                epochs=args.epochs,
                batch_size=int(run.config.batch_size),
                loss=run.config.loss,
                optimizer=run.config.optimizer,
                learning_rate=float(run.config.learning_rate),
                weight_decay=float(run.config.weight_decay),
                hidden_size=hidden_sizes,
                activation=run.config.activation,
                weight_init=run.config.weight_init,
                wandb_project=args.wandb_project,
                seed=args.seed,
            )

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
            if args.log_test_metrics:
                test_metrics = model.evaluate(X_test, y_test, batch_size=config.batch_size)

            for epoch_idx in range(len(history["train_loss"])):
                payload = {
                    "epoch": epoch_idx + 1,
                    "train_loss": history["train_loss"][epoch_idx],
                    "train_accuracy": history["train_accuracy"][epoch_idx],
                }
                if epoch_idx < len(history["val_loss"]):
                    payload["val_loss"] = history["val_loss"][epoch_idx]
                    payload["val_accuracy"] = history["val_accuracy"][epoch_idx]
                    payload["val_f1"] = history["val_f1"][epoch_idx]
                wandb.log(payload)

            run.summary["hidden_size"] = hidden_sizes
            run.summary["hidden_spec"] = run.config.hidden_spec
            run.summary["train_accuracy"] = history["train_accuracy"][-1]
            run.summary["val_accuracy"] = val_metrics["accuracy"]
            run.summary["val_f1"] = val_metrics["f1"]
            if test_metrics is not None:
                run.summary["test_accuracy"] = test_metrics["accuracy"]
                run.summary["test_f1"] = test_metrics["f1"]

    sweep_id = args.sweep_id
    if not args.analyze_only:
        sweep_id = wandb.sweep(
            sweep=sweep_config,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )
        print(f"Created sweep: {sweep_id}")
        wandb.agent(
            sweep_id,
            function=train_sweep_run,
            count=args.count,
            project=args.wandb_project,
            entity=args.wandb_entity,
        )

    if not sweep_id:
        raise ValueError("A sweep_id is required when --analyze_only is used.")

    analyze_sweep(args, sweep_id)


if __name__ == "__main__":
    main()
