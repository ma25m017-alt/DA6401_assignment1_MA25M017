from __future__ import annotations

import argparse

import numpy as np
import wandb

from common import add_run_arguments, build_bar_plot, class_names, init_wandb, load_datasets, log_figure

SIMILARITY_NOTES = {
    "mnist": "Digits such as 4 and 9, 3 and 5, and 7 and 9 often look similar in raw grayscale form. This can blur class boundaries and increase confusion for a shallow MLP.",
    "fashion": "T-shirt/top and Shirt, Pullover and Coat, plus Sandal and Sneaker can appear visually similar. That overlap can make hidden representations harder to separate.",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.1: data exploration and class distribution")
    add_run_arguments(parser, default_dataset="mnist")
    return parser.parse_args()



def main():
    args = parse_args()
    run = init_wandb(
        args,
        name=f"section_2_1_{args.dataset}",
        config={"dataset": args.dataset},
        group="section_2_1",
        job_type="analysis",
        tags=["section_2_1", args.dataset],
    )

    (X_train, y_train), _, _ = load_datasets(args.dataset, seed=args.seed)
    labels = class_names(args.dataset)
    counts = np.bincount(y_train, minlength=10)

    sample_table = wandb.Table(columns=["class_id", "class_name", "sample_id", "image"])
    for class_id in range(10):
        sample_indices = np.where(y_train == class_id)[0][:5]
        for order, sample_index in enumerate(sample_indices, start=1):
            sample_table.add_data(class_id, labels[class_id], f"{labels[class_id]}_{order}", wandb.Image(X_train[sample_index].reshape(28, 28)))
    wandb.log({"sample_images": sample_table})

    distribution_table = wandb.Table(columns=["class_name", "count"])
    for label, count in zip(labels, counts):
        distribution_table.add_data(label, int(count))
    wandb.log({"class_distribution_table": distribution_table})

    figure = build_bar_plot(labels, counts, title=f"{args.dataset.upper()} Class Distribution", y_label="Samples")
    log_figure("class_distribution_plot", figure)

    run.summary["visually_similar_classes"] = SIMILARITY_NOTES[args.dataset]
    run.summary["num_classes"] = 10
    run.summary["samples_per_class_logged"] = 5
    run.finish()

    print("Logged 5 sample images per class and class counts to W&B.")


if __name__ == "__main__":
    main()
