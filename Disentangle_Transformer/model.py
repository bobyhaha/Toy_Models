import math
import copy
import json
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


# ============================================================
# Config
# ============================================================

@dataclass
class ModelConfig:
    image_size: int = 28
    patch_size: int = 4
    in_chans: int = 1
    num_classes: int = 10

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1

    # disentangled-specific
    head_mlp_alpha: float = 0.4
    head_mlp_layers: int = 1
    aggregation: str = "sum"          # sum | mean | learned
    aggregation_scale: str = "sqrt"   # none | sqrt | n
    nonlinearity: str = "gelu"        # gelu | relu | silu

    def validate(self):
        assert self.image_size % self.patch_size == 0
        assert self.d_model % self.n_heads == 0
        assert self.aggregation in {"sum", "mean", "learned"}
        assert self.aggregation_scale in {"none", "sqrt", "n"}
        assert self.nonlinearity in {"gelu", "relu", "silu"}
        assert 0.0 <= self.head_mlp_alpha <= 1.0
        assert self.head_mlp_layers >= 1

    @property
    def d_head(self):
        return self.d_model // self.n_heads

    @property
    def num_patches(self):
        g = self.image_size // self.patch_size
        return g * g

    @property
    def baseline_mlp_proj_budget_no_bias(self):
        d = self.d_model
        return d * d + 2 * d * self.d_ff

    @property
    def d_ff_h(self):
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
        remaining = (1.0 - self.head_mlp_alpha) * self.baseline_mlp_proj_budget_no_bias
        x = remaining / max(1, 2 * self.d_model)
        return max(4, int(x))

    def _count_baseline_block_params(self):
        d = self.d_model
        qkv = d * (3 * d) + (3 * d)
        out_proj = d * d + d
        mlp = d * self.d_ff + self.d_ff + self.d_ff * d + d
        ln = 2 * (2 * d)
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
        print("Block budget summary")
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
    return {
        "gelu": nn.GELU(),
        "relu": nn.ReLU(),
        "silu": nn.SiLU(),
    }[name]


def attention(q, k, v, dropout_p=0.0, training=True):
    """
    q, k, v: [B, H, T, D]
    returns: [B, H, T, D]
    """
    D = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=training)
    return attn @ v


# ============================================================
# Patch embedding
# ============================================================

class PatchEmbed(nn.Module):
    """
    Converts image [B, C, H, W] -> patch tokens [B, N, d_model]
    """
    def __init__(self, image_size, patch_size, in_chans, d_model):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid

        self.proj = nn.Conv2d(
            in_chans,
            d_model,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.proj(x)              # [B, d_model, gh, gw]
        x = x.flatten(2).transpose(1, 2)  # [B, N, d_model]
        return x


# ============================================================
# Core blocks
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
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]

        out = attention(q, k, v, dropout_p=self.attn_drop.p, training=self.training)
        out = out.permute(0, 2, 1, 3).reshape(B, T, -1)

        x = x + self.out_proj(out)
        x = x + self.mlp(self.norm_mlp(x))
        return x


class DisentangledBlock(nn.Module):
    """
    attention -> per-head small MLP -> per-head up_proj -> add -> large MLP
    """
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

        # separate small mlp for each head
        self.small_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cfg.d_head, cfg.d_ff_h),
                get_act(cfg.nonlinearity),
                nn.Dropout(cfg.dropout),
                *sum([
                    [nn.Linear(cfg.d_ff_h, cfg.d_ff_h), get_act(cfg.nonlinearity), nn.Dropout(cfg.dropout)]
                    for _ in range(cfg.head_mlp_layers - 1)
                ], [])
            )
            for _ in range(cfg.n_heads)
        ])

        # separate up_proj for each head
        self.up_projs = nn.ModuleList([
            nn.Linear(cfg.d_ff_h, d)
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
        raise ValueError(self.agg_scale)

    def forward(self, x):
        B, T, _ = x.shape
        h = self.norm_attn(x)

        qkv = self.qkv(h).reshape(B, T, 3, self.n_heads, self.d_head)
        q, k, v = qkv.unbind(2)
        q, k, v = [t.permute(0, 2, 1, 3) for t in (q, k, v)]   # [B,H,T,d_head]

        head_out = attention(q, k, v, dropout_p=self.attn_drop.p, training=self.training)

        projected = []
        for i in range(self.n_heads):
            z = self.small_mlps[i](head_out[:, i])   # [B,T,d_ff_h]
            z = self.up_projs[i](z)                  # [B,T,d_model]
            projected.append(z)

        ups = torch.stack(projected, dim=0)          # [H,B,T,d_model]

        if self.agg == "sum":
            agg = self._scale_sum(ups.sum(0))
        elif self.agg == "mean":
            agg = ups.mean(0)
        else:
            w = F.softmax(self.agg_w, dim=0)
            agg = (ups * w[:, None, None, None]).sum(0)

        x = x + agg
        x = x + self.large_mlp(self.norm_mlp(x))
        return x


# ============================================================
# Vision Transformer classifier
# ============================================================

class VisionTransformerClassifier(nn.Module):
    def __init__(self, cfg: ModelConfig, disentangled: bool = False):
        super().__init__()
        cfg.validate()
        self.cfg = cfg

        self.patch_embed = PatchEmbed(
            image_size=cfg.image_size,
            patch_size=cfg.patch_size,
            in_chans=cfg.in_chans,
            d_model=cfg.d_model,
        )

        num_tokens = cfg.num_patches + 1
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_tokens, cfg.d_model))
        self.drop = nn.Dropout(cfg.dropout)

        block_cls = DisentangledBlock if disentangled else BaselineBlock
        self.blocks = nn.ModuleList([block_cls(cfg) for _ in range(cfg.n_layers)])

        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)   # [B,N,d]

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)   # [B,N+1,d]
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]          # CLS token
        logits = self.head(cls_out)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Data
# ============================================================

def build_dataset(name="mnist", root="./data"):
    from torchvision import datasets, transforms

    if name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_ds = datasets.MNIST(root=root, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(root=root, train=False, download=True, transform=transform)
        in_chans = 1
        image_size = 28
        num_classes = 10

    elif name == "fashionmnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_ds = datasets.FashionMNIST(root=root, train=True, download=True, transform=transform)
        test_ds = datasets.FashionMNIST(root=root, train=False, download=True, transform=transform)
        in_chans = 1
        image_size = 28
        num_classes = 10

    elif name == "cifar10":
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        in_chans = 3
        image_size = 32
        num_classes = 10

    else:
        raise ValueError(f"Unknown dataset: {name}")

    return train_ds, test_ds, in_chans, image_size, num_classes


# ============================================================
# Training
# ============================================================

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item()
            total_correct += (logits.argmax(dim=-1) == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / max(1, len(dataloader))
    acc = total_correct / max(1, total)
    return avg_loss, acc


def train_one_model(
    cfg: ModelConfig,
    disentangled: bool,
    train_ds,
    test_ds,
    n_epochs=5,
    batch_size=128,
    lr=3e-4,
    device="cpu",
    seed=42,
):
    torch.manual_seed(seed)

    model = VisionTransformerClassifier(cfg, disentangled=disentangled).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    history = {
        "train_loss": [],
        "test_loss": [],
        "test_acc": [],
    }

    for epoch in range(n_epochs):
        model.train()
        total_train_loss = 0.0

        for x, y in train_dl:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_train_loss += loss.item()

        scheduler.step()

        avg_train_loss = total_train_loss / max(1, len(train_dl))
        test_loss, test_acc = evaluate(model, test_dl, device)

        history["train_loss"].append(round(avg_train_loss, 4))
        history["test_loss"].append(round(test_loss, 4))
        history["test_acc"].append(round(test_acc, 4))

        print(
            f"epoch {epoch+1:02d}/{n_epochs} | "
            f"train_loss={avg_train_loss:.4f} | "
            f"test_loss={test_loss:.4f} | "
            f"test_acc={test_acc:.4f}"
        )

    return {
        "disentangled": disentangled,
        "n_params": model.count_params(),
        "history": history,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_test_acc": history["test_acc"][-1],
    }


def run_pair_experiment(
    cfg: ModelConfig,
    train_ds,
    test_ds,
    n_epochs=5,
    batch_size=128,
    lr=3e-4,
    device="cpu",
    seed=42,
):
    results = {}

    print("\n[Baseline]")
    cfg.print_budget()
    results["baseline"] = train_one_model(
        cfg=copy.deepcopy(cfg),
        disentangled=False,
        train_ds=train_ds,
        test_ds=test_ds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )

    print("\n[Disentangled]")
    cfg.print_budget()
    results["disentangled"] = train_one_model(
        cfg=copy.deepcopy(cfg),
        disentangled=True,
        train_ds=train_ds,
        test_ds=test_ds,
        n_epochs=n_epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=seed,
    )

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "fashionmnist", "cifar10"])
    parser.add_argument("--data_root", type=str, default="./data")

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="image_results.json")

    parser.add_argument("--patch_size", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--head_mlp_alpha", type=float, default=0.4)
    parser.add_argument("--head_mlp_layers", type=int, default=1)
    parser.add_argument("--aggregation", type=str, default="sum", choices=["sum", "mean", "learned"])
    parser.add_argument("--aggregation_scale", type=str, default="sqrt", choices=["none", "sqrt", "n"])
    parser.add_argument("--nonlinearity", type=str, default="gelu", choices=["gelu", "relu", "silu"])

    args = parser.parse_args()

    train_ds, test_ds, in_chans, image_size, num_classes = build_dataset(
        name=args.dataset,
        root=args.data_root,
    )

    cfg = ModelConfig(
        image_size=image_size,
        patch_size=args.patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_mlp_alpha=args.head_mlp_alpha,
        head_mlp_layers=args.head_mlp_layers,
        aggregation=args.aggregation,
        aggregation_scale=args.aggregation_scale,
        nonlinearity=args.nonlinearity,
    )
    cfg.validate()

    results = run_pair_experiment(
        cfg=cfg,
        train_ds=train_ds,
        test_ds=test_ds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    print("\n" + "=" * 80)
    print(f"{'Model':<18} {'Params':>12} {'Test Loss':>12} {'Test Acc':>12}")
    print("-" * 80)
    for name, r in results.items():
        print(
            f"{name:<18} "
            f"{r['n_params']:>12,} "
            f"{float(r['final_test_loss']):>12.4f} "
            f"{float(r['final_test_acc']):>12.4f}"
        )
    print("=" * 80)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()