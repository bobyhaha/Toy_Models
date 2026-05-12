import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from analyze_geometry import load_model, collect_h
from sae import SparseAutoencoder

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sin")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--sae", type=str, required=True)
    parser.add_argument("--layer", type=str, default="resid_final")
    parser.add_argument("--n_grid", type=int, default=2000)
    parser.add_argument("--top_k", type=int, default=12)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    precision = ckpt["train_cfg"].get("precision", 3)
    xs, ys, H = collect_h(model, args.task, args.device, args.layer, args.n_grid, precision)
    H = torch.tensor(H, dtype=torch.float32).to(args.device)

    sae_ckpt = torch.load(args.sae, map_location=args.device)
    sae = SparseAutoencoder(sae_ckpt["d_in"], sae_ckpt["d_hidden"]).to(args.device)
    sae.load_state_dict(sae_ckpt["sae"])
    sae.eval()

    mean = sae_ckpt["mean"].to(args.device)
    std = sae_ckpt["std"].to(args.device)
    Hn = (H - mean) / std
    x_hat, Z = sae(Hn)
    Z = Z.cpu().numpy()

    # Choose features with largest variance across x-grid
    var = Z.var(axis=0)
    top = np.argsort(var)[-args.top_k:][::-1]

    out_dir = Path("plots") / args.task / "sae"
    out_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for j in top:
        z = Z[:, j]
        if z.max() > 0:
            z = z / (z.max() + 1e-9)
        plt.plot(xs, z, label=f"latent {j}")
    plt.xlabel("x")
    plt.ylabel("normalized activation")
    plt.title(f"Top SAE latents over x for {args.task}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / "top_latents_over_x.png", dpi=160)
    plt.close()

    # Feature locality: weighted center and width in x
    rows = []
    for j in top:
        w = Z[:, j].clip(min=0)
        if w.sum() < 1e-9:
            continue
        center = float((w * xs).sum() / w.sum())
        width = float(np.sqrt((w * (xs - center)**2).sum() / w.sum()))
        rows.append((int(j), center, width, float(var[j])))

    with (out_dir / "top_latents.txt").open("w") as f:
        for j, c, w, v in rows:
            f.write(f"latent={j} center_x={c:.4f} width={w:.4f} variance={v:.6f}\n")

    print(f"Wrote SAE plots to {out_dir}")

if __name__ == "__main__":
    main()
