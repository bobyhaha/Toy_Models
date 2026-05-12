import argparse
from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from tokenizer import CharTokenizer
from data import ContinuousFunctionDataset, collate
from model import TinyTransformer

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
        n = (labels != -100).sum().item()
        total_loss += loss.item()
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)

def save_checkpoint(path, model, optimizer, step, model_cfg, train_cfg, tokenizer):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "model_cfg": model_cfg.__dict__,
        "train_cfg": train_cfg.__dict__,
        "vocab": tokenizer.itos,
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sin", choices=["identity", "square", "sin", "gaussian"])
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    model_cfg = ModelConfig()
    train_cfg = TrainConfig(steps=args.steps, seed=args.seed)

    tok = CharTokenizer()
    train_ds = ContinuousFunctionDataset(task=args.task, n=train_cfg.train_size, precision=train_cfg.precision, seed=args.seed)
    test_ds = ContinuousFunctionDataset(task=args.task, n=train_cfg.test_size, precision=train_cfg.precision, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate)

    model = TinyTransformer(
        vocab_size=tok.vocab_size,
        max_seq_len=model_cfg.max_seq_len,
        d_model=model_cfg.d_model,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        d_mlp=model_cfg.d_mlp,
        dropout=model_cfg.dropout,
    ).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    out_dir = Path("checkpoints") / args.task
    metrics_path = out_dir / "metrics.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    pbar = tqdm(total=train_cfg.steps)
    while step < train_cfg.steps:
        for batch in train_loader:
            step += 1
            input_ids = batch["input_ids"].to(args.device)
            labels = batch["labels"].to(args.device)

            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % train_cfg.eval_every == 0 or step == 1:
                test_loss = evaluate(model, test_loader, args.device)
                row = {"step": step, "train_loss": float(loss.item()), "test_loss": float(test_loss)}
                with metrics_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                pbar.set_description(f"{args.task} loss {loss.item():.4f} test {test_loss:.4f}")

            if step % train_cfg.save_every == 0 or step == 1:
                save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, step, model_cfg, train_cfg, tok)

            pbar.update(1)
            if step >= train_cfg.steps:
                break

    save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, step, model_cfg, train_cfg, tok)
    pbar.close()

if __name__ == "__main__":
    main()
