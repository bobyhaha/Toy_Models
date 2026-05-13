import argparse
import json
from pathlib import Path

import numpy as np
import torch

from interactive_pca import (
    checkpoint_matches_run,
    checkpoint_step,
    collect_h,
    load_latest_run_id,
    load_model,
    pca_frame,
    select_checkpoints,
)


def load_training_metrics(metrics_path, run_id=None):
    rows = []
    if not metrics_path.exists():
        return rows

    with metrics_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if run_id is not None and row.get("run_id") != run_id:
                continue
            rows.append(row)
    return sorted(rows, key=lambda r: r["step"])


def accuracy_for_step(metrics, step):
    if not metrics:
        return np.nan

    steps = np.array([m["step"] for m in metrics])
    if "test_token_accuracy" in metrics[0]:
        values = np.array([m["test_token_accuracy"] for m in metrics], dtype=float)
    elif "test_exact_match" in metrics[0]:
        values = np.array([m["test_exact_match"] for m in metrics], dtype=float)
    else:
        values = np.exp(-np.array([m["test_loss"] for m in metrics], dtype=float))

    return float(values[np.argmin(np.abs(steps - step))])


def x_bin_labels(xs, n_bins):
    edges = np.linspace(float(xs.min()), float(xs.max()), n_bins + 1)
    labels = np.digitize(xs, edges[1:-1], right=False)
    return labels.astype(int)


def cluster_strength(Z, labels):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return np.nan

    X = Z[:, :3]
    diff = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    scores = []

    for i, label in enumerate(labels):
        same = labels == label
        same[i] = False
        a = distances[i, same].mean() if same.any() else 0.0

        b = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            other = labels == other_label
            if other.any():
                b = min(b, distances[i, other].mean())

        denom = max(a, b)
        if np.isfinite(denom) and denom > 0:
            scores.append((b - a) / denom)

    return float(np.mean(scores)) if scores else np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sin")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--run_id", type=str, default=None, help="Run id to analyze; defaults to latest_run.txt when present.")
    parser.add_argument("--layer", type=str, default="resid_final")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_grid", type=int, default=1000)
    parser.add_argument("--n_bins", type=int, default=8, help="Number of x regions used for the cluster-strength score.")
    parser.add_argument("--max_checkpoints", type=int, default=0, help="Subsample to at most this many checkpoints; 0 means all.")
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints") / args.task
    run_id = args.run_id or load_latest_run_id(ckpt_dir)
    ckpt_paths = sorted(ckpt_dir.glob("step_*.pt"), key=checkpoint_step)
    ckpt_paths = [p for p in ckpt_paths if checkpoint_matches_run(p, run_id, args.device)]
    ckpt_paths = select_checkpoints(ckpt_paths, args.max_checkpoints)
    if not ckpt_paths:
        raise FileNotFoundError(f"No checkpoints found for task={args.task!r} run_id={run_id!r}.")

    metrics = load_training_metrics(ckpt_dir / "metrics.jsonl", run_id)
    rows = []
    previous_components = None

    for ckpt_path in ckpt_paths:
        model, ckpt = load_model(ckpt_path, args.device)
        precision = ckpt["train_cfg"].get("precision", 3)
        xs, _, H = collect_h(
            model,
            args.task,
            args.device,
            layer=args.layer,
            n_grid=args.n_grid,
            precision=precision,
        )
        Z, previous_components, evr = pca_frame(H, previous_components)
        labels = x_bin_labels(xs, args.n_bins)
        step = checkpoint_step(ckpt_path)
        if step < 0:
            step = ckpt.get("step", len(rows))
        rows.append(
            {
                "experiment": "experiment 1",
                "run_id": run_id,
                "step": step,
                "cluster_strength": cluster_strength(Z, labels),
                "test_token_accuracy": accuracy_for_step(metrics, step),
                "pca_ev_1": float(evr[0]),
                "pca_ev_2": float(evr[1]),
                "pca_ev_3": float(evr[2]),
            }
        )

    out_dir = Path("plots") / args.task
    out_dir.mkdir(parents=True, exist_ok=True)

    out_jsonl = out_dir / "experiment_1_cluster_accuracy.jsonl"
    with out_jsonl.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    steps = [r["step"] for r in rows]
    clusters = [r["cluster_strength"] for r in rows]
    accuracies = [r["test_token_accuracy"] for r in rows]

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    line1 = ax1.plot(steps, clusters, marker="o", color="tab:blue", label="cluster strength")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("cluster strength (x-bin silhouette)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    line2 = ax2.plot(steps, accuracies, marker="s", color="tab:green", label="test token accuracy")
    ax2.set_ylabel("test token accuracy", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_ylim(0, 1.02)

    lines = line1 + line2
    ax1.legend(lines, [line.get_label() for line in lines], loc="best")
    ax1.set_title(f"Experiment 1 - {args.task}: Cluster Formation vs Accuracy")
    ax1.grid(alpha=0.25)
    fig.tight_layout()

    out_png = out_dir / "experiment_1_cluster_accuracy.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print(f"Saved comparison plot to {out_png}")
    print(f"Saved comparison data to {out_jsonl}")


if __name__ == "__main__":
    main()
