from __future__ import annotations

import argparse

import wandb

from common import add_run_arguments, api_project_path, build_scatter_plot, init_wandb, log_figure



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.7: global performance analysis")
    add_run_arguments(parser, default_dataset="mnist")
    parser.add_argument("--sweep_id", default=None)
    parser.add_argument("--overfit_gap", type=float, default=0.08)
    parser.add_argument("--api_timeout", type=int, default=60)
    return parser.parse_args()



def main():
    args = parse_args()
    sweep_id = args.sweep_id.strip() if args.sweep_id else input("Enter the sweep ID from section 2.2: ").strip()
    if not sweep_id:
        raise ValueError("A sweep_id from section 2.2 is required.")

    run = init_wandb(
        args,
        name="section_2_7_global_performance_analysis",
        config={"sweep_id": sweep_id, "overfit_gap": args.overfit_gap, "api_timeout": args.api_timeout},
        group="section_2_7",
        job_type="analysis",
        tags=["section_2_7", args.dataset],
    )

    api = wandb.Api(timeout=args.api_timeout)
    project_path = api_project_path(args.wandb_entity, args.wandb_project)
    sweep = api.sweep(f"{project_path}/{sweep_id}")
    runs = sweep.runs

    train_scores = []
    test_scores = []
    labels = []
    skipped_runs = []
    summary_table = wandb.Table(columns=["run_name", "train_accuracy", "test_accuracy", "gap", "status"])

    for candidate in runs:
        try:
            summary = dict(candidate.summary_metrics)
        except Exception:
            skipped_runs.append(candidate.name)
            continue

        train_accuracy = summary.get("train_accuracy")
        test_accuracy = summary.get("test_accuracy", summary.get("final_accuracy"))
        if train_accuracy is None or test_accuracy is None:
            skipped_runs.append(candidate.name)
            continue

        gap = float(train_accuracy) - float(test_accuracy)
        status = "overfit" if gap > args.overfit_gap else "generalized"
        train_scores.append(float(train_accuracy))
        test_scores.append(float(test_accuracy))
        labels.append(candidate.name)
        summary_table.add_data(candidate.name, float(train_accuracy), float(test_accuracy), gap, status)

    if not labels:
        run.summary["source_sweep_id"] = sweep_id
        run.summary["skipped_runs"] = skipped_runs
        run.finish()
        raise RuntimeError("No sweep runs with usable train/test accuracy were found for the provided section 2.2 sweep ID.")

    wandb.log({"global_performance_table": summary_table})
    log_figure(
        "global_performance_overlay",
        build_scatter_plot(
            train_scores,
            test_scores,
            labels,
            "Training vs Test Accuracy Across Sweep Runs",
            "Training Accuracy",
            "Test Accuracy",
            threshold=args.overfit_gap,
        ),
    )

    overfit_runs = [label for label, train, test in zip(labels, train_scores, test_scores) if train - test > args.overfit_gap]
    run.summary["source_sweep_id"] = sweep_id
    run.summary["analysed_runs"] = len(labels)
    run.summary["skipped_runs"] = skipped_runs
    run.summary["high_train_low_test_runs"] = overfit_runs
    run.summary["interpretation"] = "A large train-test gap indicates overfitting: the network memorized training patterns but failed to generalize to unseen images."
    run.finish()

    print(f"Analysed {len(labels)} sweep runs from sweep {sweep_id}.")
    if skipped_runs:
        print(f"Skipped {len(skipped_runs)} runs without usable train/test metrics.")


if __name__ == "__main__":
    main()
