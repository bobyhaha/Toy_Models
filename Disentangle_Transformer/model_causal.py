import math
import json
import copy
import argparse
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Config
# ============================================================

@dataclass
class ModelConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 512
    seq_len: int = 64
    vocab_size: int = 256
    dropout: float = 0.1

    # Disentangled-specific
    head_mlp_alpha: float = 0.4
    head_mlp_layers: int = 1
    aggregation: str = "sum"          # sum | mean | learned
    aggregation_scale: str = "sqrt"   # none | sqrt | n
    nonlinearity: str = "gelu"        # gelu | relu | silu

    def validate(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.aggregation in {"sum", "mean", "learned"}
        assert self.aggregation_scale in {"none", "sqrt", "n"}
        assert self.nonlinearity in {"gelu", "relu", "silu"}
        assert 0.0 <= self.head_mlp_alpha <= 1.0
        assert self.head_mlp_layers >= 1
        assert self.seq_len >= 2
        assert self.vocab_size >= 2

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @property
    def baseline_mlp_proj_budget_no_bias(self):
        # baseline out_proj + shared MLP weights only
        # d*d + d*d_ff + d_ff*d = d^2 + 2*d*d_ff
        d = self.d_model
        return d * d + 2 * d * self.d_ff

    @property
    def d_ff_h(self):
        """
        Choose d_ff_h to approximately match the chosen fraction of the
        baseline block's projection+MLP weight budget.

        For head_mlp_layers = L:
            per-head weights =
                d_head*d_ff_h + (L-1)*d_ff_h^2 + d_ff_h*d_model
            total head weights = n_heads * per-head

        We solve approximately:
            n_heads * [(d_head + d_model)*x + (L-1)*x^2] = alpha * budget
        """
        alpha_budget = self.head_mlp_alpha * self.baseline_mlp_proj_budget_no_bias
        H = self.n_heads
        a = max(0, self.head_mlp_layers - 1) * H
        b = H * (self.d_head + self.d_model)
        c = -alpha_budget

        if a == 0:
            x = alpha_budget / max(1, b)
        else:
            disc = b * b - 4 * a * c
            x = (-b + math.sqrt(max(0.0, disc))) / (2 * a)

        return max(4, int(x))

    @property
    def d_ff_large(self):
        """
        Remaining budget goes to large MLP:
            2 * d_model * d_ff_large ≈ (1-alpha) * budget
        """
        remaining = (1.0 - self.head_mlp_alpha) * self.baseline_mlp_proj_budget_no_bias
        x = remaining / max(1, 2 * self.d_model)
        return max(4, int(x))

    def _count_baseline_block_params(self):
        d = self.d_model
        # qkv + out_proj + MLP + LN
        qkv = d * (3 * d) + (3 * d)
        out_proj = d * d + d
        mlp = d * self.d_ff + self.d_ff + self.d_ff * d + d
        ln = 2 * (2 * d)  # two layernorms: gamma+beta each
        return qkv + out_proj + mlp + ln

    def _count_disent_block_params(self):
        d = self.d_model
        dh = self.d_head
        dfh = self.d_ff_h
        dfl = self.d_ff_large

        qkv = d * (3 * d) + (3 * d)

        per_head = dh * dfh + dfh
        if self.head_mlp_layers > 1:
            per_head += (self.head_mlp_layers - 1) * (dfh * dfh + dfh)
        per_head += dfh * d + d

        heads = self.n_heads * per_head
        agg = self.n_heads if self.aggregation == "learned" else 0
        large = d * dfl + dfl + dfl * d + d
        ln = 2 * (2 * d)

        return qkv + heads + agg + large + ln

    def print_budget(self):
        b = self._count_baseline_block_params()
        d = self._count_disent_block_params()
        print(f"\n{'=' * 72}")
        print("Model budget summary")
        print(f"d_model={self.d_model}, n_heads={self.n_heads}, n_layers={self.n_layers}, d_ff={self.d_ff}")
        print(f"head_mlp_alpha={self.head_mlp_alpha}, head_mlp_layers={self.head_mlp_layers}")
        print(f"d_ff_h={self.d_ff_h}, d_ff_large={self.d_ff_large}")
        print(f"baseline block params     : {b:,}")
        print(f"disentangled block params : {d:,}")
        print(f"ratio                     : {d / b:.4f}")
        print(f"{'=' * 72}\n")


# ============================================================
# Utils
# ============================================================

def get_act(name: str) -> nn.Module:
    table = {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }
    return table[name]


def causal_attention(q, k, v, dropout_p=0.0, training=True):
    """
    q, k, v: [B, H, T, D]
    returns: [B, H, T, D]
    """
    B, H, T, D = q.shape
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, T, T]

    mask = torch.triu(
        torch.ones(T, T, device=q.device, dtype=torch.bool),
        diagonal=1
    )
    scores = scores.masked_fill(mask, float("-inf"))

    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    return attn @ v


# ============================================================
# Blocks
# ============================================================

class HeadMLP(nn.Module):
    def __init__(self, d_head, d_ff_h, d_model, n_layers=1, act="gelu", dropout=0.1):
        super().__init__()
        layers = [nn.Linear(d_head, d_ff_h), get_act(act), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d_ff_h, d_ff_h), get_act(act), nn.Dropout(dropout)]
        self.small_mlp = nn.Sequential(*layers)
        self.up_proj = nn.Linear(d_ff_h, d_model)

    def forward(self, h):
        # h: [B, T, d_head]
        return self.up_proj(self.small_mlp(h))


class BaselineBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head

        self.norm_attn = nn.LayerNorm(d)
        self.norm_mlp = nn.LayerNorm(d)

        self.qkv = nn.Linear(d, 3 * d)
        self.out_proj = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(cfg.dropout)

        self.mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff),
            get_act(cfg.nonlinearity),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, d),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        B, T, _ = x.shape

        h = self.norm_attn(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]  # [B,H,T,D]

        out = causal_attention(q, k, v, dropout_p=self.attn_drop.p, training=self.training)
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)

        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DisentangledBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_head
        self.agg = cfg.aggregation
        self.agg_scale = cfg.aggregation_scale

        self.norm_attn = nn.LayerNorm(d)
        self.norm_mlp = nn.LayerNorm(d)

        self.qkv = nn.Linear(d, 3 * d)
        self.attn_drop = nn.Dropout(cfg.dropout)

        self.head_mlps = nn.ModuleList([
            HeadMLP(
                d_head=cfg.d_head,
                d_ff_h=cfg.d_ff_h,
                d_model=d,
                n_layers=cfg.head_mlp_layers,
                act=cfg.nonlinearity,
                dropout=cfg.dropout,
            )
            for _ in range(cfg.n_heads)
        ])

        if self.agg == "learned":
            self.agg_w = nn.Parameter(torch.zeros(cfg.n_heads))

        self.large_mlp = nn.Sequential(
            nn.Linear(d, cfg.d_ff_large),
            get_act(cfg.nonlinearity),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff_large, d),
            nn.Dropout(cfg.dropout),
        )

    def _scale_sum(self, agg):
        if self.agg_scale == "none":
            return agg
        if self.agg_scale == "sqrt":
            return agg / math.sqrt(self.n_heads)
        if self.agg_scale == "n":
            return agg / self.n_heads
        raise ValueError(f"Unknown aggregation_scale={self.agg_scale}")

    def forward(self, x):
        B, T, _ = x.shape

        h = self.norm_attn(x)
        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]  # [B,H,T,D]

        head_out = causal_attention(q, k, v, dropout_p=self.attn_drop.p, training=self.training)  # [B,H,T,D]

        ups = torch.stack(
            [self.head_mlps[i](head_out[:, i]) for i in range(self.n_heads)],
            dim=0
        )  # [H,B,T,d_model]

        if self.agg == "sum":
            agg = self._scale_sum(ups.sum(0))
        elif self.agg == "mean":
            agg = ups.mean(0)
        else:
            w = F.softmax(self.agg_w, dim=0)  # [H]
            agg = (ups * w[:, None, None, None]).sum(0)

        x = x + agg
        x = x + self.large_mlp(self.norm_mlp(x))
        return x


# ============================================================
# Full LM
# ============================================================

class TransformerLM(nn.Module):
    def __init__(self, cfg: ModelConfig, disentangled: bool = False):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        block_cls = DisentangledBlock if disentangled else BaselineBlock
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layers)])

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.cfg.seq_len, f"sequence length {T} exceeds configured max {self.cfg.seq_len}"

        pos = torch.arange(T, device=idx.device)
        x = self.embed(idx) + self.pos_embed(pos)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Toy Dataset
# ============================================================

class StructuredSeqDataset(Dataset):
    """
    Structured synthetic dataset:
      - bigram transition with probability p_bigram
      - copy token from k steps back with probability p_copy
      - random token otherwise

    This actually includes the copy mechanism, unlike the original code.
    """
    def __init__(
        self,
        n_samples,
        seq_len,
        vocab_size,
        p_bigram=0.55,
        p_copy=0.25,
        copy_k=3,
        seed=42,
    ):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)

        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.table = torch.randint(0, vocab_size, (vocab_size,), generator=rng)

        seqs = []
        for _ in range(n_samples):
            s = torch.zeros(seq_len + 1, dtype=torch.long)
            s[0] = torch.randint(0, vocab_size, (1,), generator=rng)

            for t in range(seq_len):
                r = torch.rand(1, generator=rng).item()

                if t >= copy_k and r < p_copy:
                    s[t + 1] = s[t + 1 - copy_k]
                elif r < p_copy + p_bigram:
                    s[t + 1] = self.table[s[t]]
                else:
                    s[t + 1] = torch.randint(0, vocab_size, (1,), generator=rng)

            seqs.append(s)

        self.data = torch.stack(seqs)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        x = self.data[i, :-1]
        y = self.data[i, 1:]
        return x, y


# ============================================================
# Optional HF Dataset Loader
# ============================================================

class TokenBlockDataset(Dataset):
    def __init__(self, token_tensor: torch.Tensor, seq_len: int):
        super().__init__()
        self.tokens = token_tensor
        self.seq_len = seq_len
        self.block = seq_len + 1

        self.n = (len(self.tokens) - self.block) // self.block
        if self.n <= 0:
            raise ValueError("Not enough tokens to create even one block.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        start = idx * self.block
        chunk = self.tokens[start:start + self.block]
        return chunk[:-1], chunk[1:]


def build_hf_datasets(dataset_name, train_split, val_split, tokenizer_name, seq_len, text_key="text", max_train_texts=None, max_val_texts=None):
    try:
        from datasets import load_dataset
        from transformers import AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "Using --dataset hf requires `datasets` and `transformers`.\n"
            "Install with: pip install datasets transformers"
        ) from e

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def load_and_tokenize(split, max_texts=None):
        ds = load_dataset(dataset_name, split=split)
        ids = []

        n = len(ds) if max_texts is None else min(len(ds), max_texts)
        for i in range(n):
            text = ds[i][text_key]
            piece = tok.encode(text, add_special_tokens=False)
            ids.extend(piece + [tok.eos_token_id])

        return torch.tensor(ids, dtype=torch.long)

    train_ids = load_and_tokenize(train_split, max_train_texts)
    val_ids = load_and_tokenize(val_split, max_val_texts)

    train_ds = TokenBlockDataset(train_ids, seq_len=seq_len)
    val_ds = TokenBlockDataset(val_ids, seq_len=seq_len)

    return train_ds, val_ds, tok


# ============================================================
# Training / Eval
# ============================================================

def evaluate(model, dataloader, vocab_size, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total_tokens += y.numel()

    avg_loss = total_loss / max(1, len(dataloader))
    acc = total_correct / max(1, total_tokens)
    ppl = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, acc, ppl


def train_one_model(
    cfg: ModelConfig,
    disentangled: bool,
    train_ds: Dataset,
    val_ds: Dataset,
    n_epochs=5,
    batch_size=32,
    lr=3e-4,
    device="cpu",
    seed=42,
):
    torch.manual_seed(seed)

    model = TransformerLM(cfg, disentangled=disentangled).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_ppl": [],
    }

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, cfg.vocab_size), y.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_dl))
        val_loss, val_acc, val_ppl = evaluate(model, val_dl, cfg.vocab_size, device)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_ppl"].append(round(val_ppl, 4) if math.isfinite(val_ppl) else "inf")

        print(
            f"epoch {epoch + 1:02d}/{n_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_ppl={val_ppl:.4f}"
        )

    result = {
        "disentangled": disentangled,
        "n_params": model.count_params(),
        "history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "final_val_acc": history["val_acc"][-1],
        "final_val_ppl": history["val_ppl"][-1],
    }
    return result


def run_pair_experiment(
    base_cfg: ModelConfig,
    train_ds: Dataset,
    val_ds: Dataset,
    n_epochs=5,
    batch_size=32,
    lr=3e-4,
    device="cpu",
    seed=42,
):
    results = {}

    print("\n[Baseline]")
    base_cfg.print_budget()
    results["baseline"] = train_one_model(
        cfg=copy.deepcopy(base_cfg),
        disentangled=False,
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )

    print("\n[Disentangled]")
    base_cfg.print_budget()
    results["disentangled"] = train_one_model(
        cfg=copy.deepcopy(base_cfg),
        disentangled=True,
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )

    return results


# ============================================================
# Smoke Test
# ============================================================

def smoke_test(device="cpu"):
    print("Running smoke test...")
    cfg = ModelConfig(
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        seq_len=16,
        vocab_size=50,
        dropout=0.1,
        head_mlp_alpha=0.4,
        head_mlp_layers=1,
        aggregation="sum",
        aggregation_scale="sqrt",
    )
    cfg.validate()

    x = torch.randint(0, cfg.vocab_size, (8, cfg.seq_len), device=device)

    base = TransformerLM(cfg, disentangled=False).to(device)
    dis = TransformerLM(cfg, disentangled=True).to(device)

    with torch.no_grad():
        yb = base(x)
        yd = dis(x)

    assert yb.shape == (8, cfg.seq_len, cfg.vocab_size)
    assert yd.shape == (8, cfg.seq_len, cfg.vocab_size)

    print(f"baseline params     : {base.count_params():,}")
    print(f"disentangled params : {dis.count_params():,}")
    print("smoke test passed.\n")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="toy", choices=["toy", "hf"])
    parser.add_argument("--dataset_name", type=str, default="roneneldan/TinyStories")
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2")
    parser.add_argument("--text_key", type=str, default="text")
    parser.add_argument("--max_train_texts", type=int, default=None)
    parser.add_argument("--max_val_texts", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results.json")

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--head_mlp_alpha", type=float, default=0.4)
    parser.add_argument("--head_mlp_layers", type=int, default=1)
    parser.add_argument("--aggregation", type=str, default="sum", choices=["sum", "mean", "learned"])
    parser.add_argument("--aggregation_scale", type=str, default="sqrt", choices=["none", "sqrt", "n"])
    parser.add_argument("--nonlinearity", type=str, default="gelu", choices=["gelu", "relu", "silu"])

    parser.add_argument("--smoke_test", action="store_true")

    args = parser.parse_args()

    cfg = ModelConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        dropout=args.dropout,
        head_mlp_alpha=args.head_mlp_alpha,
        head_mlp_layers=args.head_mlp_layers,
        aggregation=args.aggregation,
        aggregation_scale=args.aggregation_scale,
        nonlinearity=args.nonlinearity,
    )
    cfg.validate()

    if args.smoke_test:
        smoke_test(device=args.device)
        return

    if args.dataset == "toy":
        train_ds = StructuredSeqDataset(
            n_samples=3000,
            seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            seed=args.seed,
        )
        val_ds = StructuredSeqDataset(
            n_samples=800,
            seq_len=cfg.seq_len,
            vocab_size=cfg.vocab_size,
            seed=args.seed + 1,
        )

    else:
        train_ds, val_ds, tok = build_hf_datasets(
            dataset_name=args.dataset_name,
            train_split=args.train_split,
            val_split=args.val_split,
            tokenizer_name=args.tokenizer_name,
            seq_len=cfg.seq_len,
            text_key=args.text_key,
            max_train_texts=args.max_train_texts,
            max_val_texts=args.max_val_texts,
        )
        cfg.vocab_size = tok.vocab_size

    results = run_pair_experiment(
        base_cfg=cfg,
        train_ds=train_ds,
        val_ds=val_ds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print(f"{'Model':<18} {'Params':>12} {'Val Loss':>12} {'Val Acc':>12} {'Val PPL':>12}")
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<18} "
            f"{r['n_params']:>12,} "
            f"{float(r['final_val_loss']):>12.4f} "
            f"{float(r['final_val_acc']):>12.4f} "
            f"{float(r['final_val_ppl']):>12.4f}"
        )
    print("=" * 80)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()