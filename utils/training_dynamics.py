# -*- coding: utf-8 -*-
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import torch
import tqdm
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


def read_training_dynamics(td_dir: str, id_field: str = "guid", burn_out: Optional[int] = None) -> Dict[int, Any]:
    """
    Merges training dynamics (logits and labels) across epochs from files in the given directory.

    Args:
    - td_dir (str): Directory path containing the logged training dynamics files.
    - id_field (str): The field name for instance IDs in the data.
    - burn_out (Optional[int]): Limit to the number of epochs to read.

    Returns:
    - Dict[int, Any]: A dictionary mapping instance IDs to their gold label and logits across epochs.
    """
    training_dynamics = {}

    epoch_files = [f for f in os.listdir(td_dir) if os.path.isfile(os.path.join(td_dir, f))]
    n_epochs = min(len(epoch_files), burn_out) if burn_out else len(epoch_files)
    print(f"Reading {n_epochs} files from {td_dir} ...")

    for epoch_num in tqdm.tqdm(range(n_epochs)):
        epoch_file = os.path.join(td_dir, f"dynamics_epoch_{epoch_num}.jsonl")
        if not os.path.exists(epoch_file):
            raise FileNotFoundError(f"File not found: {epoch_file}")

        # Read and process each file
        with open(epoch_file, "r") as file:
            for line in file:
                record = json.loads(line.strip())
                guid = record[id_field]
                if guid not in training_dynamics:
                    training_dynamics[guid] = {"gold": record["gold"], "logits": []}
                training_dynamics[guid]["logits"].append(record[f"logits_epoch_{epoch_num}"])

    # Filter out instances missing logits for all epochs
    training_dynamics = {guid: data for guid, data in training_dynamics.items() if len(data["logits"]) == n_epochs}

    print(f"Read training dynamics for {len(training_dynamics)} train instances.")
    return training_dynamics


def compute_training_dynamics_metrics(
    training_dynamics: Dict[int, Any], include_ci: Optional[bool] = False, burn_out: Optional[int] = np.inf
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes metrics based on training dynamics(logits for each training instance across epochs),
    including confidence, variability, correctness, forgetfulness, and threshold closeness.

    Args:
    - training_dynamics (Dict[int, Any]): A dictionary containing logits for each instance across epochs.
    - include_ci (bool): Whether to include confidence intervals in variability calculations.
    - burn_out (int): Limit the number of epochs to process.

    Returns:
    - Tuple[pd.DataFrame, pd.DataFrame]:
      1. DataFrame with computed metrics.
      2. DataFrame with epoch-wise training loss and accuracy.
    """

    # Initialize metric dictionaries
    confidence, variability, correctness, forgetfulness, threshold_closeness = {}, {}, {}, {}, {}
    print("Metrics computed: confidence, variability, correctness, forgetfulness, threshold_closeness")

    def compute_variability(confidences: np.ndarray) -> float:
        if include_ci:  # Based on prior work on active bias (https://arxiv.org/abs/1704.07433)
            return np.sqrt(np.var(confidences) * (1 + 1 / (len(confidences) - 1)))
        else:
            return np.std(confidences)

    def compute_correctness(trend: List[float]) -> float:
        """
        Returns the total number of times an example was predicted correctly.
        """
        return sum(trend)

    def compute_forgetfulness(correctness_trend: List[float]) -> int:
        """
        Computes how often an example is "forgotten" (predicted incorrectly after a correct prediction).
        [see]: https://arxiv.org/abs/1812.05159
        """
        times_forgotten = 0
        learnt = False  # Predicted correctly in the current epoch.

        for is_correct in correctness_trend:
            if learnt and not is_correct:  # <-- Forgot after learning at some print!
                times_forgotten += 1
            learnt = is_correct

        return times_forgotten if times_forgotten > 0 else 1000

    def compute_threshold_closeness(confidence: float) -> float:
        return confidence * (1 - confidence)

    num_tot_epochs = len(next(iter(training_dynamics.values()))["logits"])
    burn_out_epochs = min(burn_out, num_tot_epochs)
    print(f"Computing training dynamics. Burning out at {burn_out_epochs} of {num_tot_epochs}. ")

    logits, targets, training_accuracy = defaultdict(lambda: []), defaultdict(lambda: []), defaultdict(lambda: 0)

    # Process each training instance
    for guid, record in tqdm.tqdm(training_dynamics.items()):
        gold_label = int(record["gold"])
        correctness_trend, true_probs_trend = [], []

        for epoch_idx, epoch_logits in enumerate(record["logits"][:burn_out_epochs]):
            probs = torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            true_class_prob = float(probs[gold_label])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(epoch_logits).item()
            is_correct = prediction == gold_label
            correctness_trend.append(is_correct)

            logits[epoch_idx].append(epoch_logits)
            targets[epoch_idx].append(gold_label)
            training_accuracy[epoch_idx] += is_correct

        # Compute metrics
        confidence[guid] = np.mean(true_probs_trend)
        variability[guid] = compute_variability(np.array(true_probs_trend))
        correctness[guid] = compute_correctness(correctness_trend)
        forgetfulness[guid] = compute_forgetfulness(correctness_trend)
        threshold_closeness[guid] = compute_threshold_closeness(confidence[guid])

    # Compile results into DataFrame
    metrics_df = pd.DataFrame(
        [
            {
                "guid": guid,
                "index": idx,
                "confidence": confidence[guid],
                "variability": variability[guid],
                "correctness": correctness[guid],
                "forgetfulness": forgetfulness[guid],
                "threshold_closeness": threshold_closeness[guid],
            }
            for idx, guid in enumerate(confidence)
        ]
    )

    # Compute per-epoch loss and accuracy
    loss_fn = torch.nn.CrossEntropyLoss()
    for epoch_idx in range(num_tot_epochs):
        print(torch.Tensor(logits[epoch_idx]))
        print(torch.LongTensor(targets[epoch_idx]))
        loss = loss_fn(torch.Tensor(logits[epoch_idx]), torch.LongTensor(targets[epoch_idx]))
        print(loss)

    train_metrics_df = pd.DataFrame(
        [
            {
                "epoch": epoch_idx,
                "loss": loss_fn(torch.Tensor(logits[epoch_idx]), torch.LongTensor(targets[epoch_idx])).item(),
                "accuracy": training_accuracy[epoch_idx] / len(training_dynamics),
            }
            for epoch_idx in range(num_tot_epochs)
        ],
    )
    return metrics_df, train_metrics_df


def plot_data_map(df: pd.DataFrame, hue_metric: str = "correct.", title: str = "", show_hist: bool = False) -> None:

    sns.set_theme(style="whitegrid", font_scale=1.6, context="paper")

    # Subsample data to plot, so the plot is not too busy.
    df = df.sample(n=min(25000, len(df)))

    # Normalize correctness to a value between 0 and 1.
    df["correct."] = df["correctness"] / df["correctness"].max()

    main_metric = "variability"
    other_metric = "confidence"
    num_hues = len(df[hue_metric].unique())
    style = hue_metric if num_hues < 8 else None

    if show_hist:
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, height_ratios=[5, 2])
        ax0 = fig.add_subplot(gs[0, :])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[1, 2])
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))

    # カラーパレットの選択
    palette = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    # 散布図の描画
    sns.scatterplot(
        x=main_metric,
        y=other_metric,
        ax=ax0,
        data=df,
        hue=hue_metric,
        palette=palette,
        style=style,
        s=30,
    )

    # アノテーションの追加
    annotations = [
        ("ambiguous", (0.9, 0.5), "black"),
        ("easy-to-learn", (0.27, 0.85), "r"),
        ("hard-to-learn", (0.35, 0.25), "b"),
    ]
    for text, xy, color in annotations:
        ax0.annotate(
            text,
            xy=xy,
            xycoords="axes fraction",
            fontsize=15,
            color="black",
            va="center",
            ha="center",
            rotation=350 if color == "black" else 0,
            bbox=dict(boxstyle="round,pad=0.3", ec=color, lw=2, fc="white"),
        )

    if show_hist:
        ax0.legend(ncol=1, fancybox=True, shadow=True)
    else:
        ax0.legend(ncol=1, loc="upper right")
    ax0.set_xlabel("variability")
    ax0.set_ylabel("confidence")
    ax0.set_title(f"Data Map {title}", fontsize=16)

    if show_hist:
        histgrams = [
            (ax1, "confidence", "#622a87", "confidence"),
            (ax2, "variability", "teal", "variability"),
            (ax3, "correct.", "#86bf91", "correctness"),
        ]
        for ax, col, color, xlabel in histgrams:
            df.hist(column=[col], ax=ax, color=color, grid=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density" if col != "correct." else "")
            ax.set_title("")

    fig.tight_layout()
    plt.show()
    return None
