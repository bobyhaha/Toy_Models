import argparse
import json
from pathlib import Path
import re

import numpy as np
import plotly.graph_objects as go
import torch

from data import make_grid_prompts
from model import TinyTransformer
from tokenizer import CharTokenizer


def checkpoint_step(path):
    m = re.search(r"step_(\d+)\.pt", str(path))
    return int(m.group(1)) if m else -1


def select_checkpoints(paths, max_checkpoints):
    if max_checkpoints <= 0 or len(paths) <= max_checkpoints:
        return paths
    idx = np.linspace(0, len(paths) - 1, max_checkpoints).round().astype(int)
    return [paths[i] for i in sorted(set(idx))]


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


def load_latest_run_id(ckpt_dir):
    latest_path = ckpt_dir / "latest_run.txt"
    if latest_path.exists():
        return latest_path.read_text().strip() or None
    return None


def checkpoint_matches_run(path, run_id, device):
    if run_id is None:
        return True
    ckpt = torch.load(path, map_location=device)
    return ckpt.get("run_id") == run_id


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    tok = CharTokenizer()
    cfg = ckpt["model_cfg"]
    model = TinyTransformer(
        vocab_size=tok.vocab_size,
        max_seq_len=cfg["max_seq_len"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"],
        d_mlp=cfg["d_mlp"],
        dropout=cfg["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def collect_h(model, task, device, layer="resid_final", n_grid=1000, precision=3):
    xs, ys, input_ids, last_pos = make_grid_prompts(task, n=n_grid, precision=precision)
    input_ids = input_ids.to(device)
    last_pos = last_pos.to(device)
    all_h = []
    bs = 256
    for i in range(0, input_ids.size(0), bs):
        inp = input_ids[i:i + bs]
        pos = last_pos[i:i + bs]
        _, cache = model(inp, return_cache=True)
        h = cache[layer]
        h_last = h[torch.arange(h.size(0), device=device), pos]
        all_h.append(h_last.cpu())
    return xs, ys, torch.cat(all_h).numpy()


def pca_frame(H, previous_components=None):
    Hc = H - H.mean(axis=0, keepdims=True)
    _, singular_values, components = np.linalg.svd(Hc, full_matrices=False)
    components = components[:3]
    Z = Hc @ components.T
    explained_variance = singular_values ** 2 / max(H.shape[0] - 1, 1)
    total_variance = explained_variance.sum()
    if total_variance > 0:
        explained_variance_ratio = explained_variance[:3] / total_variance
    else:
        explained_variance_ratio = np.zeros(3)

    # Keep component signs stable from frame to frame so the slider does not
    # appear to jump because PCA chose the opposite direction for an axis.
    if previous_components is not None:
        for i in range(components.shape[0]):
            if np.dot(previous_components[i], components[i]) < 0:
                components[i] *= -1
                Z[:, i] *= -1

    return Z, components, explained_variance_ratio


def umap_frame(H, n_neighbors=15, min_dist=0.1, random_state=0):
    try:
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires the umap-learn package. "
            "Install it with: pip install umap-learn"
        ) from exc

    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )
    Z = reducer.fit_transform(H)
    return Z, np.array([np.nan, np.nan, np.nan])


def project_frame(H, method, previous_components=None, umap_neighbors=15, umap_min_dist=0.1, random_state=0):
    if method == "pca":
        return pca_frame(H, previous_components)
    if method == "umap":
        Z, evr = umap_frame(H, umap_neighbors, umap_min_dist, random_state)
        return Z, previous_components, evr
    raise ValueError(f"Unknown projection method: {method}")


def nearest_neighbor_summaries(H, xs, k):
    if k <= 0:
        return np.array([""] * len(xs), dtype=object), np.empty((len(xs), 0), dtype=int)

    diff = H[:, None, :] - H[None, :, :]
    distances = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(distances, np.inf)
    neighbor_idx = np.argsort(distances, axis=1)[:, :k]
    summaries = []
    for i, row in enumerate(neighbor_idx):
        parts = [f"x={xs[j]:.6g} (d={distances[i, j]:.3g})" for j in row]
        summaries.append("<br>".join(parts))
    return np.array(summaries, dtype=object), neighbor_idx


def make_trace(Z, xs, ys, step, accuracy, neighbor_text, method, visible=True):
    customdata = np.column_stack([
        xs,
        ys,
        np.full(len(xs), accuracy),
        neighbor_text,
    ])
    axis1, axis2, axis3 = projection_axis_labels(method)
    return go.Scatter3d(
        x=Z[:, 0],
        y=Z[:, 1],
        z=Z[:, 2],
        mode="markers",
        marker={
            "size": 3,
            "color": xs,
            "colorscale": "Viridis",
            "colorbar": {"title": "x"} if visible else None,
            "showscale": visible,
        },
        customdata=customdata,
        hovertemplate=(
            "x = %{customdata[0]:.6g}<br>"
            "y = %{customdata[1]:.6g}<br>"
            "accuracy = %{customdata[2]:.3%}<br>"
            f"{axis1} = %{{x:.6g}}<br>"
            f"{axis2} = %{{y:.6g}}<br>"
            f"{axis3} = %{{z:.6g}}<br>"
            "nearest neighbors:<br>%{customdata[3]}"
            f"<extra>step {step}</extra>"
        ),
        name=f"step {step}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sin")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional single checkpoint to plot.")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory containing step_*.pt checkpoints.")
    parser.add_argument("--run_id", type=str, default=None, help="Checkpoint run id to plot; defaults to latest_run.txt when present.")
    parser.add_argument("--method", type=str, default="pca", choices=["pca", "umap"])
    parser.add_argument("--max_checkpoints", type=int, default=0, help="Subsample to at most this many checkpoints; 0 means all.")
    parser.add_argument("--layer", type=str, default="resid_final")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_grid", type=int, default=1000)
    parser.add_argument("--knn", type=int, default=5, help="Number of nearest neighbors to show on hover.")
    parser.add_argument("--umap_neighbors", type=int, default=15)
    parser.add_argument("--umap_min_dist", type=float, default=0.1)
    parser.add_argument("--umap_random_state", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ckpt_dir = None
    if args.checkpoint:
        ckpt_paths = [Path(args.checkpoint)]
        run_id = args.run_id
    else:
        ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else Path("checkpoints") / args.task
        run_id = args.run_id or load_latest_run_id(ckpt_dir)
        ckpt_paths = sorted(ckpt_dir.glob("step_*.pt"), key=checkpoint_step)
        ckpt_paths = [p for p in ckpt_paths if checkpoint_matches_run(p, run_id, args.device)]

    ckpt_paths = select_checkpoints(ckpt_paths, args.max_checkpoints)
    if not ckpt_paths:
        raise FileNotFoundError(
            f"No checkpoints found. Pass --checkpoint or put step_*.pt files under checkpoints/{args.task}."
        )

    if ckpt_dir is None:
        ckpt_dir = ckpt_paths[0].parent
    metrics = load_training_metrics(ckpt_dir / "metrics.jsonl", run_id)

    frame_data = []
    previous_components = None
    global_min = np.array([np.inf, np.inf, np.inf])
    global_max = np.array([-np.inf, -np.inf, -np.inf])

    for ckpt_path in ckpt_paths:
        model, ckpt = load_model(ckpt_path, args.device)
        precision = ckpt["train_cfg"].get("precision", 3)

        xs, ys, H = collect_h(
            model,
            args.task,
            args.device,
            layer=args.layer,
            n_grid=args.n_grid,
            precision=precision,
        )

        step = checkpoint_step(ckpt_path)
        if step < 0:
            step = ckpt.get("step", len(frame_data))
        Z, previous_components, evr = project_frame(
            H,
            args.method,
            previous_components=previous_components,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            random_state=args.umap_random_state,
        )
        neighbor_text, neighbor_idx = nearest_neighbor_summaries(H, xs, args.knn)
        accuracy = accuracy_for_step(metrics, step)
        frame_data.append({
            "step": step,
            "xs": xs,
            "ys": ys,
            "Z": Z,
            "evr": evr,
            "accuracy": accuracy,
            "neighbor_text": neighbor_text,
            "neighbor_idx": neighbor_idx,
        })
        global_min = np.minimum(global_min, Z.min(axis=0))
        global_max = np.maximum(global_max, Z.max(axis=0))

    first = frame_data[0]
    fig = go.Figure(
        data=[
            make_trace(
                first["Z"],
                first["xs"],
                first["ys"],
                first["step"],
                first["accuracy"],
                first["neighbor_text"],
                args.method,
            )
        ],
        frames=[
            go.Frame(
                data=[
                    make_trace(
                        d["Z"],
                        d["xs"],
                        d["ys"],
                        d["step"],
                        d["accuracy"],
                        d["neighbor_text"],
                        args.method,
                        visible=False,
                    )
                ],
                name=str(d["step"]),
                layout={"title": frame_title(args.task, d, args.method)},
            )
            for d in frame_data
        ],
    )

    sliders = [
        {
            "active": 0,
            "currentvalue": {"prefix": "training step: "},
            "pad": {"t": 45},
            "steps": [
                {
                    "label": str(d["step"]),
                    "method": "animate",
                    "args": [
                        [str(d["step"])],
                        {
                            "mode": "immediate",
                            "frame": {"duration": 0, "redraw": True},
                            "transition": {"duration": 0},
                        },
                    ],
                }
                for d in frame_data
            ],
        }
    ]

    axis_padding = np.maximum((global_max - global_min) * 0.05, 1e-6)
    x_range = [global_min[0] - axis_padding[0], global_max[0] + axis_padding[0]]
    y_range = [global_min[1] - axis_padding[1], global_max[1] + axis_padding[1]]
    z_range = [global_min[2] - axis_padding[2], global_max[2] + axis_padding[2]]
    axis1, axis2, axis3 = projection_axis_labels(args.method)
    fig.update_layout(
        title=frame_title(args.task, first, args.method),
        scene={
            "xaxis": {"title": axis1, "range": x_range},
            "yaxis": {"title": axis2, "range": y_range},
            "zaxis": {"title": axis3, "range": z_range},
        },
        margin={"l": 0, "r": 0, "t": 70, "b": 0},
        sliders=sliders,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 0,
                "xanchor": "left",
                "yanchor": "top",
                "pad": {"t": 65, "r": 10},
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {"duration": 500, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": True},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
    )

    out_dir = Path("plots") / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = f"interactive_{args.method}.html"
    out_path = Path(args.output) if args.output else out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path)
    print(f"Saved interactive {args.method.upper()} over {len(frame_data)} checkpoint(s) to {out_path}")


def projection_axis_labels(method):
    if method == "pca":
        return "PC1", "PC2", "PC3"
    if method == "umap":
        return "UMAP1", "UMAP2", "UMAP3"
    raise ValueError(f"Unknown projection method: {method}")


def frame_title(task, frame, method):
    evr = frame["evr"]
    accuracy = frame["accuracy"]
    acc_text = "accuracy unavailable" if np.isnan(accuracy) else f"accuracy: {accuracy:.1%}"
    if method == "pca":
        detail = f"explained variance: {evr[0]:.1%}, {evr[1]:.1%}, {evr[2]:.1%}"
    else:
        detail = "3D UMAP projection"
    return f"Experiment 1 - {task}: interactive {method.upper()} of h(x), step {frame['step']} ({acc_text}; {detail})"

if __name__ == "__main__":
    main()
