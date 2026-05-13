import argparse
from pathlib import Path
import json
from datetime import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import ModelConfig, TrainConfig
from tokenizer import CharTokenizer
from data import TASKS, ContinuousFunctionDataset, collate
from model import TinyTransformer

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    exact_sequences = 0
    total_sequences = 0
    per_task_correct = {task: 0 for task in TASKS}
    per_task_tokens = {task: 0 for task in TASKS}
    per_task_exact = {task: 0 for task in TASKS}
    per_task_sequences = {task: 0 for task in TASKS}
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        task_ids = batch["task_id"].to(device)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum")
        mask = labels != -100
        preds = logits.argmax(dim=-1)
        correct_by_pos = (preds == labels) & mask
        exact_by_seq = ((preds == labels) | ~mask).all(dim=1)
        n = mask.sum().item()
        total_loss += loss.item()
        total_tokens += n
        correct_tokens += correct_by_pos.sum().item()
        exact_sequences += exact_by_seq.sum().item()
        total_sequences += labels.size(0)
        for task_id, task in enumerate(TASKS):
            task_mask = task_ids == task_id
            if not task_mask.any():
                continue
            per_task_correct[task] += correct_by_pos[task_mask].sum().item()
            per_task_tokens[task] += mask[task_mask].sum().item()
            per_task_exact[task] += exact_by_seq[task_mask].sum().item()
            per_task_sequences[task] += task_mask.sum().item()
    model.train()
    return {
        "loss": total_loss / max(total_tokens, 1),
        "token_accuracy": correct_tokens / max(total_tokens, 1),
        "exact_match": exact_sequences / max(total_sequences, 1),
        "per_task_token_accuracy": {
            task: per_task_correct[task] / max(per_task_tokens[task], 1)
            for task in TASKS
        },
        "per_task_exact_match": {
            task: per_task_exact[task] / max(per_task_sequences[task], 1)
            for task in TASKS
        },
    }

def save_checkpoint(path, model, optimizer, step, model_cfg, train_cfg, tokenizer, run_id):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "run_id": run_id,
        "model_cfg": model_cfg.__dict__,
        "train_cfg": train_cfg.__dict__,
        "vocab": tokenizer.itos,
    }, path)


def should_run_interval(
    step,
    early_until,
    early_every,
    mid_until,
    mid_every,
    regular_every,
):
    if step == 1:
        return True
    if step <= early_until:
        return step % early_every == 0
    if step <= mid_until:
        return step % mid_every == 0
    return step % regular_every == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="sin", choices=TASKS + ["all"])
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
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text("")
    (out_dir / "latest_run.txt").write_text(run_id)

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

            if should_run_interval(
                step,
                train_cfg.early_steps,
                train_cfg.early_eval_every,
                train_cfg.mid_steps,
                train_cfg.mid_eval_every,
                train_cfg.eval_every,
            ):
                test_metrics = evaluate(model, test_loader, args.device)
                row = {
                    "run_id": run_id,
                    "experiment": "experiment 1",
                    "step": step,
                    "train_loss": float(loss.item()),
                    "test_loss": float(test_metrics["loss"]),
                    "test_token_accuracy": float(test_metrics["token_accuracy"]),
                    "test_exact_match": float(test_metrics["exact_match"]),
                    "per_task_token_accuracy": test_metrics["per_task_token_accuracy"],
                    "per_task_exact_match": test_metrics["per_task_exact_match"],
                }
                with metrics_path.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                pbar.set_description(
                    f"{args.task} loss {loss.item():.4f} "
                    f"test {test_metrics['loss']:.4f} acc {test_metrics['token_accuracy']:.3f}"
                )

            if should_run_interval(
                step,
                train_cfg.early_steps,
                train_cfg.early_save_every,
                train_cfg.mid_steps,
                train_cfg.mid_save_every,
                train_cfg.save_every,
            ):
                save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, step, model_cfg, train_cfg, tok, run_id)

            pbar.update(1)
            if step >= train_cfg.steps:
                break

    save_checkpoint(out_dir / f"step_{step}.pt", model, optimizer, step, model_cfg, train_cfg, tok, run_id)
    pbar.close()

if __name__ == "__main__":
    main()
