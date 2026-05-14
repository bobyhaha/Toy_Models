import argparse
from pathlib import Path
import json
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.stats import spearmanr

from checkpoint_io import load_checkpoint
from tokenizer import CharTokenizer
from data import make_grid_prompts
from model import TinyTransformer

def load_model(ckpt_path, device):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
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
def collect_h(model, task, device, layer="resid_final", n_grid=1000, precision=3, condition_task=None):
    xs, ys, input_ids, last_pos = make_grid_prompts(
        task,
        n=n_grid,
        precision=precision,
        condition_task=condition_task,
    )
    input_ids = input_ids.to(device)
    last_pos = last_pos.to(device)
    all_h = []
    bs = 256
    for i in range(0, input_ids.size(0), bs):
        inp = input_ids[i:i+bs]
        pos = last_pos[i:i+bs]
        logits, cache = model(inp, return_cache=True)
        h = cache[layer]
        h_last = h[torch.arange(h.size(0), device=device), pos]
        all_h.append(h_last.cpu())
    return xs, ys, torch.cat(all_h).numpy()

def participation_ratio(evals):
    evals = np.maximum(evals, 0)
    return (evals.sum() ** 2) / (np.square(evals).sum() + 1e-12)

def geometry_metrics(xs, ys, H):
    Hc = H - H.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(20, H.shape[1]))
    Z = pca.fit_transform(Hc)
    ev = pca.explained_variance_

    # local smoothness: average distance between neighboring x-grid points
    local = np.linalg.norm(np.diff(H, axis=0), axis=1).mean()

    # distance correlation with |x_i-x_j| and |y_i-y_j|
    idx = np.linspace(0, len(xs)-1, min(300, len(xs))).astype(int)
    DH = pairwise_distances(H[idx], metric="euclidean")
    DX = np.abs(xs[idx, None] - xs[None, idx])
    DY = np.abs(ys[idx, None] - ys[None, idx])
    triu = np.triu_indices_from(DH, k=1)
    corr_x = spearmanr(DH[triu], DX[triu]).correlation
    corr_y = spearmanr(DH[triu], DY[triu]).correlation

    return {
        "local_smoothness": float(local),
        "spearman_dist_x": float(corr_x),
        "spearman_dist_y": float(corr_y),
        "pca_dim_pr": float(participation_ratio(ev)),
        "pca_ev_1": float(pca.explained_variance_ratio_[0]),
        "pca_ev_2": float(pca.explained_variance_ratio_[1]) if len(pca.explained_variance_ratio_) > 1 else 0.0,
        "Z2": Z[:, :2],
    }

def checkpoint_step(path):
    m = re.search(r"step_(\d+)\.pt", str(path))
    return int(m.group(1)) if m else -1


def load_latest_run_id(ckpt_dir):
    latest_path = ckpt_dir / "latest_run.txt"
    if latest_path.exists():
        return latest_path.read_text().strip() or None
    return None


def checkpoint_matches_run(path, run_id, device):
    if run_id is None:
        return True
    ckpt = load_checkpoint(path, map_location=device)
    return ckpt.get("run_id") == run_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sin")
    parser.add_argument("--condition_task", type=str, default=None, help="Function label to condition on when --task all.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer", type=str, default="resid_final")
    parser.add_argument("--n_grid", type=int, default=1000)
    args = parser.parse_args()

    ckpt_dir = Path("checkpoints") / args.task
    run_id = load_latest_run_id(ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("step_*.pt"), key=checkpoint_step)
    ckpts = [p for p in ckpts if checkpoint_matches_run(p, run_id, args.device)]
    out_dir = Path("plots") / args.task
    if args.task == "all" and args.condition_task:
        out_dir = out_dir / args.condition_task
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    selected_steps = set()
    if ckpts:
        selected_steps = {checkpoint_step(ckpts[0]), checkpoint_step(ckpts[len(ckpts)//2]), checkpoint_step(ckpts[-1])}

    for ckpt_path in ckpts:
        model, ckpt = load_model(ckpt_path, args.device)
        precision = ckpt["train_cfg"].get("precision", 3)
        xs, ys, H = collect_h(
            model,
            args.task,
            args.device,
            args.layer,
            args.n_grid,
            precision,
            condition_task=args.condition_task,
        )
        m = geometry_metrics(xs, ys, H)
        step = checkpoint_step(ckpt_path)
        rows.append({k: v for k, v in m.items() if k != "Z2"} | {"step": step})

        if step in selected_steps:
            Z2 = m["Z2"]
            plt.figure(figsize=(6, 5))
            sc = plt.scatter(Z2[:, 0], Z2[:, 1], c=xs, s=8)
            plt.colorbar(sc, label="x")
            plt.title(f"{args.task} PCA of h(x), step {step}")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.tight_layout()
            plt.savefig(out_dir / f"pca_step_{step}.png", dpi=160)
            plt.close()

    with (out_dir / "geometry_metrics.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if rows:
        steps = [r["step"] for r in rows]
        for key in ["local_smoothness", "spearman_dist_x", "spearman_dist_y", "pca_dim_pr", "pca_ev_1"]:
            plt.figure(figsize=(6,4))
            plt.plot(steps, [r[key] for r in rows], marker="o")
            plt.xlabel("training step")
            plt.ylabel(key)
            plt.title(f"{args.task}: {key}")
            plt.tight_layout()
            plt.savefig(out_dir / f"{key}.png", dpi=160)
            plt.close()

    print(f"Wrote plots and metrics to {out_dir}")

if __name__ == "__main__":
    main()
