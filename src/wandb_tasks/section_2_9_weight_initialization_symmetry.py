from __future__ import annotations

import argparse

import numpy as np
import wandb

from common import add_run_arguments, build_line_plot, init_wandb, load_datasets, log_figure, make_config, zero_like_weights
from ann.neural_network import NeuralNetwork



def parse_args():
    parser = argparse.ArgumentParser(description="Section 2.9: weight initialization and symmetry")
    add_run_arguments(parser, default_dataset="mnist")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    parser.add_argument("--max_steps", type=int, default=50)
    return parser.parse_args()



def collect_gradient_curves(config, X_train, y_train, max_steps: int, init_mode: str):
    model = NeuralNetwork(config)
    if init_mode == "zeros":
        model.set_weights(zero_like_weights(model))

    curves = {f"neuron_{index}": [] for index in range(5)}
    losses = []
    step = 0
    for X_batch, y_batch in model._iterate_minibatches(X_train, y_train, config.batch_size, shuffle=False):
        logits = model.forward(X_batch)
        loss = model.loss_function(logits, y_batch) + model._l2_penalty()
        model.backward(y_batch, logits)
        for neuron_idx in range(5):
            curves[f"neuron_{neuron_idx}"] .append(float(np.linalg.norm(model.layers[0].grad_W[:, neuron_idx])))
        losses.append(float(loss))
        model.update_weights()
        step += 1
        if step >= max_steps:
            break
    return curves, losses



def main():
    args = parse_args()
    run = init_wandb(
        args,
        name="section_2_9_weight_initialization_symmetry",
        config={"batch_size": args.batch_size, "learning_rate": args.learning_rate, "max_steps": args.max_steps},
        group="section_2_9",
        job_type="analysis",
        tags=["section_2_9", args.dataset],
    )

    (X_train, y_train), _, _ = load_datasets(args.dataset, seed=args.seed)
    config = make_config(
        dataset=args.dataset,
        epochs=1,
        batch_size=args.batch_size,
        loss="cross_entropy",
        optimizer="sgd",
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        hidden_size=[64, 64],
        activation="relu",
        weight_init="xavier",
        wandb_project=args.wandb_project,
        seed=args.seed,
    )

    zero_curves, zero_losses = collect_gradient_curves(config, X_train, y_train, args.max_steps, init_mode="zeros")
    xavier_curves, xavier_losses = collect_gradient_curves(config, X_train, y_train, args.max_steps, init_mode="xavier")

    log_figure("zero_init_gradient_plot", build_line_plot(zero_curves, "Zero Initialization: Gradient Curves for 5 Neurons", "Iteration", "Gradient Norm"))
    log_figure("xavier_init_gradient_plot", build_line_plot(xavier_curves, "Xavier Initialization: Gradient Curves for 5 Neurons", "Iteration", "Gradient Norm"))
    log_figure("initialization_loss_plot", build_line_plot({"zeros": zero_losses, "xavier": xavier_losses}, "Loss over First 50 Iterations", "Iteration", "Loss"))

    summary_table = wandb.Table(columns=["initialization", "mean_loss", "final_loss", "gradient_overlap_explanation"])
    summary_table.add_data("zeros", float(np.mean(zero_losses)), float(zero_losses[-1]), "All neurons in a layer receive identical gradients, so they stay copies of each other and cannot specialize.")
    summary_table.add_data("xavier", float(np.mean(xavier_losses)), float(xavier_losses[-1]), "Distinct initial weights break symmetry, giving neurons different gradients and letting them learn different features.")
    wandb.log({"initialization_summary": summary_table})

    run.summary["symmetry_breaking_needed"] = "Without different initial values, each neuron in the layer follows the exact same update path and the MLP collapses to redundant units."
    run.finish()

    print("Logged zero-vs-Xavier gradient plots.")


if __name__ == "__main__":
    main()
