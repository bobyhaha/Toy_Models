import argparse
from pathlib import Path
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from analyze_geometry import load_model, collect_h
from sae import SparseAutoencoder

def step_from_path(p):
    m = re.search(r"step_(\d+)\.pt", str(p))
    return int(m.group(1)) if m else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sin")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--layer", type=str, default="resid_final")
    parser.add_argument("--n_grid", type=int, default=10000)
    parser.add_argument("--hidden_mult", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint, args.device)
    precision = ckpt["train_cfg"].get("precision", 3)
    xs, ys, H = collect_h(model, args.task, args.device, args.layer, args.n_grid, precision)
    H = torch.tensor(H, dtype=torch.float32)
    mean = H.mean(dim=0, keepdim=True)
    std = H.std(dim=0, keepdim=True).clamp_min(1e-6)
    Hn = (H - mean) / std

    ds = TensorDataset(Hn)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    d_in = Hn.size(1)
    sae = SparseAutoencoder(d_in, args.hidden_mult * d_in).to(args.device)
    opt = torch.optim.AdamW(sae.parameters(), lr=args.lr)

    it = iter(loader)
    pbar = tqdm(range(args.steps))
    for step in pbar:
        try:
            (x,) = next(it)
        except StopIteration:
            it = iter(loader)
            (x,) = next(it)
        x = x.to(args.device)
        loss, info = sae.loss(x, args.l1)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            pbar.set_description(f"loss {loss.item():.4f} mse {info['mse']:.4f} l1 {info['l1']:.4f}")

    out = Path("checkpoints") / f"{args.task}_sae_step_{step_from_path(args.checkpoint)}.pt"
    torch.save({
        "sae": sae.state_dict(),
        "d_in": d_in,
        "d_hidden": args.hidden_mult * d_in,
        "mean": mean,
        "std": std,
        "task": args.task,
        "checkpoint": args.checkpoint,
        "layer": args.layer,
    }, out)
    print(f"Saved SAE to {out}")

if __name__ == "__main__":
    main()
