"""
SAE Lens Manifold-Refined SAE Experiment with Controls
======================================================

This script tests whether SAE decoder vectors behave like means of local
feature manifolds.

Main reconstruction:

    SAE:
        x_hat = b + sum_i f_i(x) v_i

    Manifold model:
        x_hat = b + sum_{i in topk(f)} f_i(x) [v_i + delta_i(x)]
        delta_i(x) = U_i z_i(x)

where:
    v_i is interpreted as the feature mean direction mu_i
    delta_i(x) is the context-dependent deviation from that mean.

Controls included:
    1. Scalar-rescaling baseline:
        x_hat = b + sum_i alpha_i(x) f_i(x) v_i

    2. Same-size generic residual MLP adversary:
        x_hat = x_hat_topk + g(x_hat_full, x_hat_topk)

    3. Random activation controls:
        --random-control gaussian
        --random-control shuffle_dims

    4. Manifold context ablation:
        --context-mode ids_only
            delta_i sees only f_i and v_i. This is the strictest test.

        --context-mode sae_recon
            delta_i sees f_i, v_i, and an SAE reconstruction context.
            By default this context is x_hat_topk; use
            --sae-context-source full to reproduce the old x_hat_full context.

        --context-mode raw_x
            delta_i sees f_i, v_i, and the original activation x.
            If only this mode works, the model is probably learning a direct
            residual predictor x - x_hat rather than feature-local structure.

Install:
    pip install torch transformer-lens sae-lens datasets tqdm

Example run:
    python model.py \
        --dataset tinystories \
        --max-examples 20000 \
        --num-epochs 5 \
        --max-token-blocks 8192 \
        --activation-batch-size 512 \
        --top-k 64 \
        --rank 4 \
        --lr 1e-4 \
        --lambda-delta 1e-2 \
        --lambda-z 1e-3 \
        --lambda-parallel 1e-2 \
        --log-every 100

Random control:
    python model.py \
        --dataset tinystories \
        --random-control shuffle_dims \
        --max-examples 20000 \
        --num-epochs 5 \
        --max-token-blocks 8192 \
        --activation-batch-size 512 \
        --top-k 64 \
        --rank 4 \
        --lr 1e-4 \
        --lambda-delta 1e-2 \
        --lambda-z 1e-3 \
        --lambda-parallel 1e-2 \
        --log-every 100
"""

import argparse
import gc
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer


# ============================================================
# 0. Utilities
# ============================================================


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_float32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(torch.float32)


def scalar(x) -> float:
    if torch.is_tensor(x):
        return x.detach().float().item()
    return float(x)


def format_metrics(metrics: Dict[str, float]) -> str:
    return " ".join(f"{k}={v:.6g}" for k, v in metrics.items())


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def print_param_count(name: str, model: nn.Module) -> None:
    trainable = count_trainable_params(model)
    total = count_all_params(model)
    print(f"{name} params: trainable={trainable:,}, total={total:,}")


# ============================================================
# 1. Text loading
# ============================================================


def load_tinyshakespeare_text() -> str:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urllib.request.urlopen(url) as f:
        return f.read().decode("utf-8")


def load_tinystories_texts(
    split: str = "train",
    max_examples: Optional[int] = None,
) -> List[str]:
    ds = load_dataset("roneneldan/TinyStories", split=split)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    return [ex["text"] for ex in ds]


def get_texts(dataset: str, max_examples: Optional[int]) -> List[str]:
    name = dataset.lower()

    if name in {"tinyshakespeare", "shakespeare"}:
        return [load_tinyshakespeare_text()]

    if name in {"tinystories", "tiny_stories", "stories"}:
        return load_tinystories_texts(split="train", max_examples=max_examples)

    raise ValueError(f"Unknown dataset: {dataset}")


class TokenBlockDataset(Dataset):
    """
    Converts raw text into fixed-length token blocks using TransformerLens tokenizer.
    """

    def __init__(
        self,
        model: HookedTransformer,
        texts: List[str],
        seq_len: int,
        max_tokens: Optional[int] = None,
    ):
        self.seq_len = seq_len

        token_chunks: List[torch.Tensor] = []
        total_tokens = 0

        for text in tqdm(texts, desc="Tokenizing"):
            toks = model.to_tokens(text, prepend_bos=False).squeeze(0).cpu()
            token_chunks.append(toks)
            total_tokens += toks.numel()

            if max_tokens is not None and total_tokens >= max_tokens:
                break

        if len(token_chunks) == 0:
            raise ValueError("No text was tokenized.")

        all_tokens = torch.cat(token_chunks, dim=0)

        if max_tokens is not None:
            all_tokens = all_tokens[:max_tokens]

        n_blocks = all_tokens.numel() // seq_len
        all_tokens = all_tokens[: n_blocks * seq_len]

        if n_blocks == 0:
            raise ValueError(
                f"No token blocks created. Got {all_tokens.numel()} tokens with seq_len={seq_len}."
            )

        self.tokens = all_tokens.view(n_blocks, seq_len).long()

        print(
            f"TokenBlockDataset: tokens={all_tokens.numel()}, "
            f"seq_len={seq_len}, blocks={len(self.tokens)}"
        )

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


# ============================================================
# 2. Activation datasets and random controls
# ============================================================


class ActivationDataset(Dataset):
    """
    Materializes activations from a TransformerLens hook point into memory.
    """

    def __init__(
        self,
        tl_model: HookedTransformer,
        token_dataset: Dataset,
        hook_name: str,
        token_batch_size: int,
        device: str,
        max_token_blocks: Optional[int] = None,
        token_position: str = "all",
    ):
        self.hook_name = hook_name
        self.token_position = token_position

        if max_token_blocks is not None:
            n = min(max_token_blocks, len(token_dataset))
            token_dataset = torch.utils.data.Subset(token_dataset, range(n))

        token_loader = DataLoader(
            token_dataset,
            batch_size=token_batch_size,
            shuffle=False,
            drop_last=False,
        )

        acts_cpu: List[torch.Tensor] = []

        tl_model.eval()

        with torch.no_grad():
            for tokens in tqdm(token_loader, desc=f"Caching activations at {hook_name}"):
                tokens = tokens.to(device)

                _, cache = tl_model.run_with_cache(
                    tokens,
                    names_filter=[hook_name],
                )

                if hook_name not in cache:
                    available = list(cache.keys())
                    raise KeyError(
                        f"Hook name {hook_name} not found in cache. "
                        f"Available cached keys: {available[:20]}"
                    )

                acts = cache[hook_name]

                # Usually [batch, seq_len, d_model].
                if token_position == "all":
                    acts = acts.reshape(-1, acts.shape[-1])
                elif token_position == "last":
                    acts = acts[:, -1, :]
                elif token_position.startswith("index:"):
                    idx = int(token_position.split(":", 1)[1])
                    acts = acts[:, idx, :]
                else:
                    raise ValueError(f"Unknown token_position: {token_position}")

                acts_cpu.append(to_float32(acts).cpu())

        if len(acts_cpu) == 0:
            raise RuntimeError("No activations were cached.")

        self.activations = torch.cat(acts_cpu, dim=0)

        print(f"ActivationDataset: activations={tuple(self.activations.shape)}")

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.activations[idx]


class RandomActivationDataset(Dataset):
    """
    Random controls for overfitting/capacity testing.

    Modes:
        gaussian:
            random Gaussian vectors with the same per-dimension mean/std
            as real activations.

        shuffle_dims:
            independently shuffle each dimension across examples.
            This preserves per-coordinate marginals but destroys vector geometry.

        permute_examples:
            only permutes examples. This should behave like real data and is a sanity check.
    """

    def __init__(
        self,
        reference_dataset: Dataset,
        seed: int = 0,
        mode: str = "gaussian",
    ):
        xs = []

        for i in range(len(reference_dataset)):
            x = reference_dataset[i]
            if isinstance(x, (tuple, list)):
                x = x[0]
            xs.append(x.float().cpu())

        X = torch.stack(xs, dim=0)

        self.mode = mode
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True).clamp_min(1e-6)

        g = torch.Generator().manual_seed(seed)

        if mode == "gaussian":
            self.activations = self.mean + self.std * torch.randn(
                X.shape,
                generator=g,
            )

        elif mode == "shuffle_dims":
            X_rand = X.clone()
            for d in tqdm(range(X.shape[1]), desc="Shuffling activation dimensions"):
                perm = torch.randperm(X.shape[0], generator=g)
                X_rand[:, d] = X_rand[perm, d]
            self.activations = X_rand

        elif mode == "permute_examples":
            perm = torch.randperm(X.shape[0], generator=g)
            self.activations = X[perm]

        else:
            raise ValueError(f"Unknown random mode: {mode}")

        print(
            f"RandomActivationDataset: mode={mode}, "
            f"shape={tuple(self.activations.shape)}"
        )

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]


def make_activation_loaders(
    tl_model: HookedTransformer,
    dataset: str,
    hook_name: str,
    seq_len: int,
    token_batch_size: int,
    activation_batch_size: int,
    device: str,
    max_examples: Optional[int],
    max_tokens: Optional[int],
    max_token_blocks: Optional[int],
    val_fraction: float,
    seed: int,
    random_control: str = "none",
) -> Tuple[DataLoader, DataLoader]:
    texts = get_texts(dataset, max_examples=max_examples)

    token_dataset = TokenBlockDataset(
        model=tl_model,
        texts=texts,
        seq_len=seq_len,
        max_tokens=max_tokens,
    )

    act_dataset = ActivationDataset(
        tl_model=tl_model,
        token_dataset=token_dataset,
        hook_name=hook_name,
        token_batch_size=token_batch_size,
        device=device,
        max_token_blocks=max_token_blocks,
        token_position="all",
    )

    if random_control != "none":
        act_dataset = RandomActivationDataset(
            reference_dataset=act_dataset,
            seed=seed,
            mode=random_control,
        )

    if len(act_dataset) < 2:
        raise RuntimeError("Need at least 2 activation vectors to create train/val split.")

    n_val = max(1, int(len(act_dataset) * val_fraction))
    n_train = len(act_dataset) - n_val

    if n_train <= 0:
        raise RuntimeError(
            f"Train split is empty: total={len(act_dataset)}, val={n_val}, train={n_train}."
        )

    generator = torch.Generator().manual_seed(seed)

    train_set, val_set = random_split(
        act_dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_batch_size = min(activation_batch_size, max(1, len(train_set)))
    val_batch_size = min(activation_batch_size, max(1, len(val_set)))

    print(
        f"Activation dataset sizes: total={len(act_dataset)}, "
        f"train={len(train_set)}, val={len(val_set)}, "
        f"train_batch_size={train_batch_size}, val_batch_size={val_batch_size}"
    )

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        drop_last=False,
    )

    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    if len(train_loader) == 0:
        raise RuntimeError(
            "train_loader has zero batches. Increase --max-token-blocks, "
            "increase --max-tokens, or reduce --activation-batch-size."
        )

    return train_loader, val_loader


# ============================================================
# 3. SAE Lens wrapper
# ============================================================


class FrozenSAEWrapper(nn.Module):
    """
    Compatibility wrapper.

    We want:
        encode(x) -> feature activations
        W_dec -> [d_sae, d_model]
        b_dec -> [d_model]
    """

    def __init__(self, sae: SAE):
        super().__init__()
        self.sae = sae

        for p in self.sae.parameters():
            p.requires_grad = False

    @property
    def W_dec(self) -> torch.Tensor:
        return self.sae.W_dec

    @property
    def b_dec(self) -> torch.Tensor:
        if not hasattr(self.sae, "b_dec"):
            raise AttributeError("This SAE does not expose b_dec.")
        return self.sae.b_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        if hasattr(self.sae, "decode"):
            return self.sae.decode(f)
        return self.b_dec + f @ self.W_dec


# ============================================================
# 4. Common SAE utilities
# ============================================================


def get_topk_sae_quantities(
    sae: FrozenSAEWrapper,
    x: torch.Tensor,
    top_k: int,
    normalize_decoder: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Computes frozen SAE features and full/top-k reconstructions.

    Returns:
        f: [batch, d_sae]
        f_active: [batch, k]
        active_idx: [batch, k]
        v_active: [batch, k, d_model]
        x_hat_sae_full: [batch, d_model]
        x_hat_sae_topk: [batch, d_model]
    """
    W_dec = sae.W_dec
    b_dec = sae.b_dec

    if normalize_decoder:
        W_use = F.normalize(W_dec, dim=-1)
    else:
        W_use = W_dec

    with torch.no_grad():
        f = sae.encode(x)

        x_hat_sae_full = b_dec + f @ W_use

        k = min(top_k, f.shape[-1])
        f_active, active_idx = torch.topk(
            f,
            k=k,
            dim=-1,
            largest=True,
            sorted=False,
        )

        v_active = W_use[active_idx]

        x_hat_sae_topk = b_dec + (
            f_active[..., None] * v_active
        ).sum(dim=1)

    return {
        "f": f,
        "f_active": f_active,
        "active_idx": active_idx,
        "v_active": v_active,
        "x_hat_sae_full": x_hat_sae_full,
        "x_hat_sae_topk": x_hat_sae_topk,
    }


# ============================================================
# 5. Scalar-rescaling baseline
# ============================================================


class ScalarRescaleSAE(nn.Module):
    """
    Baseline:
        x_hat = b + sum_i alpha_i(x) f_i(x) v_i

    This tests whether the improvement is just amplitude correction.
    """

    def __init__(
        self,
        sae: FrozenSAEWrapper,
        d_model: int,
        top_k: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.sae = sae
        self.d_model = d_model
        self.top_k = top_k

        for p in self.sae.parameters():
            p.requires_grad = False

        in_dim = 1 + d_model + d_model

        self.alpha_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Start as alpha = 1.
        nn.init.zeros_(self.alpha_mlp[-1].weight)
        nn.init.zeros_(self.alpha_mlp[-1].bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = get_topk_sae_quantities(
            sae=self.sae,
            x=x,
            top_k=self.top_k,
            normalize_decoder=False,
        )

        f_active = q["f_active"]
        v_active = q["v_active"]
        x_hat_sae_full = q["x_hat_sae_full"]
        x_hat_sae_topk = q["x_hat_sae_topk"]

        batch, k, d_model = v_active.shape
        x_context = x_hat_sae_full[:, None, :].expand(batch, k, d_model)

        inp = torch.cat(
            [
                f_active[..., None],
                v_active,
                x_context,
            ],
            dim=-1,
        )

        alpha = 1.0 + 0.1 * self.alpha_mlp(inp).squeeze(-1)

        x_hat_rescaled = self.sae.b_dec + (
            alpha[..., None] * f_active[..., None] * v_active
        ).sum(dim=1)

        return {
            **q,
            "x_hat_rescaled": x_hat_rescaled,
            "alpha": alpha,
        }

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)

        x_hat_rescaled = out["x_hat_rescaled"]
        x_hat_sae_full = out["x_hat_sae_full"]
        x_hat_sae_topk = out["x_hat_sae_topk"]

        loss = F.mse_loss(x_hat_rescaled, x)

        with torch.no_grad():
            loss_sae_full = F.mse_loss(x_hat_sae_full, x)
            loss_sae_topk = F.mse_loss(x_hat_sae_topk, x)

            rel_improve_vs_topk = (
                loss_sae_topk - loss
            ) / loss_sae_topk.clamp_min(1e-8)

            rel_improve_vs_full = (
                loss_sae_full - loss
            ) / loss_sae_full.clamp_min(1e-8)

            mean_l0 = (out["f"] > 0).float().sum(dim=-1).mean()

            alpha_abs_mean = (out["alpha"] - 1.0).abs().mean()
            alpha_std = out["alpha"].std()

        logs = {
            "loss": loss.detach(),
            "loss_sae_full": loss_sae_full.detach(),
            "loss_sae_topk": loss_sae_topk.detach(),
            "rel_improve_vs_topk": rel_improve_vs_topk.detach(),
            "rel_improve_vs_full": rel_improve_vs_full.detach(),
            "mean_l0": mean_l0.detach(),
            "alpha_abs_mean": alpha_abs_mean.detach(),
            "alpha_std": alpha_std.detach(),
        }

        return loss, logs


# ============================================================
# 6. Manifold-refined SAE
# ============================================================


@dataclass
class ManifoldSAEConfig:
    d_model: int
    d_sae: int
    top_k: int = 32
    rank: int = 4
    hidden_dim: int = 256

    lambda_delta: float = 1e-3
    lambda_z: float = 1e-4
    lambda_parallel: float = 1e-3

    normalize_decoder: bool = False
    use_offset_scale: bool = True
    offset_scale_init: float = -5.0

    # What information the offset network is allowed to see.
    #   ids_only:  f_i and v_i only. No global x-like context.
    #   sae_recon: f_i, v_i, and SAE reconstruction context.
    #   raw_x:     f_i, v_i, and original activation x. This is the leaky control.
    context_mode: str = "sae_recon"

    # For context_mode=sae_recon, choose whether the context vector is
    # x_hat_sae_topk or x_hat_sae_full. "topk" is stricter; "full" reproduces
    # the old behavior of the original script.
    sae_context_source: str = "topk"


class OffsetMLP(nn.Module):
    """
    Predicts low-dimensional coordinate z_i for each active feature.

    This is the key context-ablation module.

    context_mode="ids_only":
        z_i = h(f_i, v_i)
        The offset predictor sees no raw x and no SAE reconstruction context.
        This is the strictest test of whether individual active feature identity
        and coefficient are enough to improve reconstruction.

    context_mode="sae_recon":
        z_i = h(f_i, v_i, x_hat_sae_context)
        The offset predictor sees only SAE-derived information, not raw x.
        The context source is controlled by cfg.sae_context_source.

    context_mode="raw_x":
        z_i = h(f_i, v_i, x)
        This is the leaky/high-information control. If this works much better
        than the other two modes, the model is likely learning a generic
        residual correction from x.
    """

    def __init__(
        self,
        d_model: int,
        rank: int,
        hidden_dim: int,
        context_mode: str,
    ):
        super().__init__()

        if context_mode not in {"ids_only", "sae_recon", "raw_x"}:
            raise ValueError(
                "context_mode must be one of: ids_only, sae_recon, raw_x; "
                f"got {context_mode!r}"
            )

        self.context_mode = context_mode

        # Always include the scalar coefficient f_i and the decoder vector v_i.
        # Add a d_model-dimensional context only for sae_recon/raw_x.
        context_dim = 0 if context_mode == "ids_only" else d_model
        in_dim = 1 + d_model + context_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rank),
        )

        # Start at zero correction.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        f_active: torch.Tensor,
        v_active: torch.Tensor,
        x_hat_sae_context: Optional[torch.Tensor] = None,
        raw_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, k, d_model = v_active.shape
        f_context = f_active[..., None]

        if self.context_mode == "ids_only":
            inp = torch.cat([f_context, v_active], dim=-1)

        elif self.context_mode == "sae_recon":
            if x_hat_sae_context is None:
                raise ValueError("context_mode='sae_recon' requires x_hat_sae_context")
            context = x_hat_sae_context[:, None, :].expand(batch, k, d_model)
            inp = torch.cat([f_context, v_active, context], dim=-1)

        elif self.context_mode == "raw_x":
            if raw_x is None:
                raise ValueError("context_mode='raw_x' requires raw_x")
            context = raw_x[:, None, :].expand(batch, k, d_model)
            inp = torch.cat([f_context, v_active, context], dim=-1)

        else:
            raise RuntimeError(f"Unexpected context_mode: {self.context_mode}")

        return self.net(inp)


class ManifoldRefinedSAE(nn.Module):
    """
    Frozen SAE plus feature-conditioned low-rank local offsets.

    Standard SAE:
        x_hat = b + sum_i f_i(x) v_i

    Manifold model:
        x_hat = b + sum_{i in topk} f_i(x) [v_i + delta_i(x)]
        delta_i(x) = U_i z_i(x)

    Here v_i is interpreted as mu_i, the mean direction of feature manifold i.
    """

    def __init__(self, sae: FrozenSAEWrapper, cfg: ManifoldSAEConfig):
        super().__init__()

        self.sae = sae
        self.cfg = cfg

        for p in self.sae.parameters():
            p.requires_grad = False

        self.offset_mlp = OffsetMLP(
            d_model=cfg.d_model,
            rank=cfg.rank,
            hidden_dim=cfg.hidden_dim,
            context_mode=cfg.context_mode,
        )

        # Feature-specific local basis.
        # Shape: [d_sae, d_model, rank].
        self.U = nn.Parameter(
            0.01 * torch.randn(cfg.d_sae, cfg.d_model, cfg.rank)
        )

        if cfg.use_offset_scale:
            self.offset_scale = nn.Parameter(torch.tensor(cfg.offset_scale_init))
        else:
            self.offset_scale = None

    def get_offset_scale(self) -> torch.Tensor:
        if self.offset_scale is None:
            return torch.tensor(1.0, device=self.U.device)
        return torch.sigmoid(self.offset_scale)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        q = get_topk_sae_quantities(
            sae=self.sae,
            x=x,
            top_k=cfg.top_k,
            normalize_decoder=cfg.normalize_decoder,
        )

        f_active = q["f_active"]
        active_idx = q["active_idx"]
        v_active = q["v_active"]
        x_hat_sae_full = q["x_hat_sae_full"]
        x_hat_sae_topk = q["x_hat_sae_topk"]

        if cfg.context_mode == "sae_recon":
            if cfg.sae_context_source == "topk":
                x_hat_sae_context = x_hat_sae_topk
            elif cfg.sae_context_source == "full":
                x_hat_sae_context = x_hat_sae_full
            else:
                raise ValueError(
                    "sae_context_source must be one of: topk, full; "
                    f"got {cfg.sae_context_source!r}"
                )
        else:
            x_hat_sae_context = None

        z = self.offset_mlp(
            f_active=f_active,
            v_active=v_active,
            x_hat_sae_context=x_hat_sae_context,
            raw_x=x if cfg.context_mode == "raw_x" else None,
        )

        U_active = self.U[active_idx]

        delta_raw = torch.einsum("bkdr,bkr->bkd", U_active, z)
        scale = self.get_offset_scale()
        delta = scale * delta_raw

        refined_atoms = v_active + delta

        x_hat_refined = self.sae.b_dec + (
            f_active[..., None] * refined_atoms
        ).sum(dim=1)

        return {
            **q,
            "x_hat_refined": x_hat_refined,
            "z": z,
            "delta": delta,
            "delta_raw": delta_raw,
            "offset_scale_value": scale.detach(),
        }

    def compute_delta_metrics(self, out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        f_active = out["f_active"]
        v_active = out["v_active"]
        delta = out["delta"]

        eps = 1e-8

        delta_norm = delta.norm(dim=-1)
        v_norm = v_active.norm(dim=-1).clamp_min(eps)

        ratio = delta_norm / v_norm

        # Weighted by f_i^2 because high-activation features matter more.
        weights = f_active.pow(2)
        weighted_delta_sq = (weights * delta_norm.pow(2)).sum()
        weighted_v_sq = (weights * v_norm.pow(2)).sum().clamp_min(eps)
        weighted_delta_over_mu = torch.sqrt(weighted_delta_sq / weighted_v_sq)

        # Fraction of delta parallel to v.
        parallel_component = (delta * v_active).sum(dim=-1) / v_norm
        parallel_norm_ratio = parallel_component.abs() / delta_norm.clamp_min(eps)

        cosine_delta_mu = (delta * v_active).sum(dim=-1) / (
            delta_norm.clamp_min(eps) * v_norm
        )

        return {
            "delta_over_mu_mean": ratio.mean(),
            "delta_over_mu_median": ratio.median(),
            "delta_over_mu_max": ratio.max(),
            "weighted_delta_over_mu": weighted_delta_over_mu,
            "delta_norm_mean": delta_norm.mean(),
            "mu_norm_mean": v_norm.mean(),
            "parallel_fraction_mean": parallel_norm_ratio.mean(),
            "cos_delta_mu_mean": cosine_delta_mu.mean(),
        }

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)

        x_hat_refined = out["x_hat_refined"]
        x_hat_sae_full = out["x_hat_sae_full"]
        x_hat_sae_topk = out["x_hat_sae_topk"]

        f_active = out["f_active"]
        v_active = out["v_active"]
        delta = out["delta"]
        z = out["z"]

        cfg = self.cfg

        loss_rec = F.mse_loss(x_hat_refined, x)

        loss_delta = (f_active[..., None].pow(2) * delta.pow(2)).mean()
        loss_z = z.pow(2).mean()

        v_norm = v_active.norm(dim=-1).clamp_min(1e-8)
        parallel_component = (delta * v_active).sum(dim=-1) / v_norm
        loss_parallel = parallel_component.pow(2).mean()

        loss = (
            loss_rec
            + cfg.lambda_delta * loss_delta
            + cfg.lambda_z * loss_z
            + cfg.lambda_parallel * loss_parallel
        )

        with torch.no_grad():
            loss_sae_full = F.mse_loss(x_hat_sae_full, x)
            loss_sae_topk = F.mse_loss(x_hat_sae_topk, x)

            residual_before = (x - x_hat_sae_topk).pow(2).mean().clamp_min(1e-8)
            residual_after = (x - x_hat_refined).pow(2).mean()
            residual_r2_vs_topk = 1.0 - residual_after / residual_before

            rel_improve_vs_topk = (
                loss_sae_topk - loss_rec
            ) / loss_sae_topk.clamp_min(1e-8)

            rel_improve_vs_full = (
                loss_sae_full - loss_rec
            ) / loss_sae_full.clamp_min(1e-8)

            mean_l0 = (out["f"] > 0).float().sum(dim=-1).mean()

            delta_metrics = self.compute_delta_metrics(out)

        logs = {
            "loss": loss.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_sae_full": loss_sae_full.detach(),
            "loss_sae_topk": loss_sae_topk.detach(),
            "loss_delta": loss_delta.detach(),
            "loss_z": loss_z.detach(),
            "loss_parallel": loss_parallel.detach(),
            "residual_r2_vs_topk": residual_r2_vs_topk.detach(),
            "rel_improve_vs_topk": rel_improve_vs_topk.detach(),
            "rel_improve_vs_full": rel_improve_vs_full.detach(),
            "mean_l0": mean_l0.detach(),
            "offset_scale": out["offset_scale_value"].detach(),
            **{k: v.detach() for k, v in delta_metrics.items()},
        }

        return loss, logs


# ============================================================
# 7. Same-size generic residual MLP adversary
# ============================================================


def mlp_param_count(d_in: int, d_hidden: int, d_out: int, n_hidden_layers: int) -> int:
    """
    Count params for:
        Linear(d_in -> h)
        n_hidden_layers times Linear(h -> h)
        Linear(h -> d_out)
    """
    total = d_in * d_hidden + d_hidden
    for _ in range(n_hidden_layers):
        total += d_hidden * d_hidden + d_hidden
    total += d_hidden * d_out + d_out
    return total


def find_hidden_dim_for_target_params(
    d_in: int,
    d_out: int,
    target_params: int,
    n_hidden_layers: int,
    max_hidden: int = 20000,
) -> int:
    """
    Finds the hidden dimension whose parameter count is closest to target_params.
    """
    best_h = 1
    best_diff = float("inf")

    lo, hi = 1, max_hidden

    # Binary-ish search to get near target.
    while lo <= hi:
        mid = (lo + hi) // 2
        c = mlp_param_count(d_in, mid, d_out, n_hidden_layers)
        diff = abs(c - target_params)

        if diff < best_diff:
            best_diff = diff
            best_h = mid

        if c < target_params:
            lo = mid + 1
        else:
            hi = mid - 1

    # Local refine.
    for h in range(max(1, best_h - 10), min(max_hidden, best_h + 10) + 1):
        c = mlp_param_count(d_in, h, d_out, n_hidden_layers)
        diff = abs(c - target_params)
        if diff < best_diff:
            best_diff = diff
            best_h = h

    return best_h


class CapacityMatchedResidualMLP(nn.Module):
    """
    Same-size adversary baseline.

    It predicts a generic residual vector rather than feature-local deltas:

        x_hat = x_hat_sae_topk + g(x_hat_sae_full, x_hat_sae_topk)

    This tests whether the manifold result is just from adding ~same number of
    trainable parameters.

    It has no explicit feature-wise delta_i = U_i z_i geometry.
    """

    def __init__(
        self,
        sae: FrozenSAEWrapper,
        d_model: int,
        top_k: int,
        target_params: int,
        n_hidden_layers: int = 1,
        hidden_dim: Optional[int] = None,
        residual_scale_init: float = -5.0,
    ):
        super().__init__()

        self.sae = sae
        self.d_model = d_model
        self.top_k = top_k
        self.n_hidden_layers = n_hidden_layers

        for p in self.sae.parameters():
            p.requires_grad = False

        d_in = 2 * d_model
        d_out = d_model

        if hidden_dim is None:
            hidden_dim = find_hidden_dim_for_target_params(
                d_in=d_in,
                d_out=d_out,
                target_params=target_params,
                n_hidden_layers=n_hidden_layers,
            )

        self.hidden_dim = hidden_dim

        layers: List[nn.Module] = []
        layers.append(nn.Linear(d_in, hidden_dim))
        layers.append(nn.GELU())

        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_dim, d_out))

        self.net = nn.Sequential(*layers)

        # Start near zero residual.
        last_linear = self.net[-1]
        assert isinstance(last_linear, nn.Linear)
        nn.init.zeros_(last_linear.weight)
        nn.init.zeros_(last_linear.bias)

        self.residual_scale = nn.Parameter(torch.tensor(residual_scale_init))

        actual_params = count_trainable_params(self)
        print(
            f"CapacityMatchedResidualMLP: target_params={target_params:,}, "
            f"hidden_dim={hidden_dim:,}, actual_trainable_params={actual_params:,}"
        )

    def get_residual_scale(self) -> torch.Tensor:
        return torch.sigmoid(self.residual_scale)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        q = get_topk_sae_quantities(
            sae=self.sae,
            x=x,
            top_k=self.top_k,
            normalize_decoder=False,
        )

        x_hat_sae_full = q["x_hat_sae_full"]
        x_hat_sae_topk = q["x_hat_sae_topk"]

        inp = torch.cat([x_hat_sae_full, x_hat_sae_topk], dim=-1)

        residual_raw = self.net(inp)
        scale = self.get_residual_scale()
        residual = scale * residual_raw

        x_hat_residual = x_hat_sae_topk + residual

        return {
            **q,
            "x_hat_residual": x_hat_residual,
            "residual": residual,
            "residual_raw": residual_raw,
            "residual_scale_value": scale.detach(),
        }

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)

        x_hat_residual = out["x_hat_residual"]
        x_hat_sae_full = out["x_hat_sae_full"]
        x_hat_sae_topk = out["x_hat_sae_topk"]

        loss = F.mse_loss(x_hat_residual, x)

        with torch.no_grad():
            loss_sae_full = F.mse_loss(x_hat_sae_full, x)
            loss_sae_topk = F.mse_loss(x_hat_sae_topk, x)

            rel_improve_vs_topk = (
                loss_sae_topk - loss
            ) / loss_sae_topk.clamp_min(1e-8)

            rel_improve_vs_full = (
                loss_sae_full - loss
            ) / loss_sae_full.clamp_min(1e-8)

            residual_norm = out["residual"].norm(dim=-1)
            topk_norm = x_hat_sae_topk.norm(dim=-1).clamp_min(1e-8)
            residual_over_topk = (residual_norm / topk_norm).mean()

            mean_l0 = (out["f"] > 0).float().sum(dim=-1).mean()

        logs = {
            "loss": loss.detach(),
            "loss_sae_full": loss_sae_full.detach(),
            "loss_sae_topk": loss_sae_topk.detach(),
            "rel_improve_vs_topk": rel_improve_vs_topk.detach(),
            "rel_improve_vs_full": rel_improve_vs_full.detach(),
            "residual_over_topk_mean": residual_over_topk.detach(),
            "residual_scale": out["residual_scale_value"].detach(),
            "mean_l0": mean_l0.detach(),
        }

        return loss, logs


# ============================================================
# 8. Training / evaluation
# ============================================================


def move_batch_to_device(batch, device: str) -> torch.Tensor:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    if isinstance(batch, (tuple, list)):
        return batch[0].to(device)

    if isinstance(batch, dict):
        if "x" in batch:
            return batch["x"].to(device)
        if "activation" in batch:
            return batch["activation"].to(device)

    raise ValueError("Unknown batch format.")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict[str, float]:
    model.eval()

    totals: Dict[str, float] = {}
    n = 0

    for batch in loader:
        x = move_batch_to_device(batch, device).float()

        _, logs = model.compute_loss(x)

        for k, v in logs.items():
            val = scalar(v)
            totals[k] = totals.get(k, 0.0) + val

        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    lr: float,
    weight_decay: float,
    num_epochs: int,
    device: str,
    log_every: int,
    save_path: Optional[str] = None,
    model_name: str = "model",
) -> nn.Module:
    if len(train_loader) == 0:
        raise RuntimeError("train_loader has zero batches.")

    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters found.")

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay,
    )

    step = 0
    best_val = math.inf

    print_param_count(model_name, model)

    for epoch in range(num_epochs):
        model.train()

        for batch in train_loader:
            x = move_batch_to_device(batch, device).float()

            loss, logs = model.compute_loss(x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            if step % log_every == 0:
                metric_dict = {
                    k: scalar(v)
                    for k, v in logs.items()
                }
                print(
                    f"[{model_name}] "
                    f"epoch={epoch} step={step} "
                    f"{format_metrics(metric_dict)}"
                )

            step += 1

        if val_loader is not None:
            metrics = evaluate_model(model, val_loader, device=device)
            print(f"[{model_name} val epoch={epoch}] {format_metrics(metrics)}")

            val_loss = metrics.get("loss", math.inf)

            if save_path is not None and val_loss < best_val:
                best_val = val_loss
                save_dir = os.path.dirname(save_path)

                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

                torch.save(model.state_dict(), save_path)
                print(f"[{model_name}] Saved best model to {save_path}")

    return model


# ============================================================
# 9. Loading model / SAE
# ============================================================


def load_model_and_sae(
    model_name: str,
    release: str,
    sae_id: str,
    device: str,
) -> Tuple[HookedTransformer, FrozenSAEWrapper, Dict]:
    print(f"Loading TransformerLens model: {model_name}")
    tl_model = HookedTransformer.from_pretrained(model_name, device=device)
    tl_model.eval()

    print(f"Loading SAE Lens SAE: release={release}, sae_id={sae_id}")
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,
        sae_id=sae_id,
        device=device,
    )
    sae.eval()

    wrapped_sae = FrozenSAEWrapper(sae)

    return tl_model, wrapped_sae, cfg_dict


def infer_dims(sae: FrozenSAEWrapper) -> Tuple[int, int]:
    W_dec = sae.W_dec

    if W_dec.ndim != 2:
        raise ValueError(f"Expected W_dec to be 2D, got shape {tuple(W_dec.shape)}")

    d_sae, d_model = W_dec.shape

    return d_sae, d_model


def infer_hook_name(sae: FrozenSAEWrapper, fallback_sae_id: str) -> str:
    hook_name = getattr(sae.sae.cfg, "hook_name", None)

    if hook_name is None:
        hook_name = fallback_sae_id

    return hook_name


# ============================================================
# 10. Summary logic
# ============================================================


def summarize_experiment(
    args: argparse.Namespace,
    scalar_metrics: Optional[Dict[str, float]],
    residual_metrics: Optional[Dict[str, float]],
    manifold_metrics: Optional[Dict[str, float]],
    train_batches: int,
    val_batches: int,
    scalar_params: Optional[int],
    residual_params: Optional[int],
    manifold_params: Optional[int],
) -> None:
    print("\n" + "=" * 90)
    print("CONCRETE EXPERIMENT SUMMARY")
    print("=" * 90)

    print("\nSetup:")
    print(f"  dataset:              {args.dataset}")
    print(f"  random_control:       {args.random_control}")
    print(f"  model_name:           {args.model_name}")
    print(f"  release:              {args.release}")
    print(f"  sae_id:               {args.sae_id}")
    print(f"  top_k:                {args.top_k}")
    print(f"  rank:                 {args.rank}")
    print(f"  context_mode:         {args.context_mode}")
    print(f"  sae_context_source:   {args.sae_context_source}")
    print(f"  num_epochs:           {args.num_epochs}")
    print(f"  lr:                   {args.lr}")
    print(f"  train_batches:        {train_batches}")
    print(f"  val_batches:          {val_batches}")

    print("\nTrainable parameter counts:")
    if scalar_params is not None:
        print(f"  scalar baseline:      {scalar_params:,}")
    if residual_params is not None:
        print(f"  residual adversary:   {residual_params:,}")
    if manifold_params is not None:
        print(f"  manifold model:       {manifold_params:,}")

    print("\nFinal validation metrics:")
    header = (
        "model              | loss       | vs_full    | vs_topk    | "
        "sae_full  | sae_topk"
    )
    print(header)
    print("-" * len(header))

    def print_row(name: str, m: Optional[Dict[str, float]]) -> None:
        if m is None:
            return
        print(
            f"{name:<18} | "
            f"{m.get('loss', float('nan')):>10.6g} | "
            f"{m.get('rel_improve_vs_full', float('nan')):>10.6g} | "
            f"{m.get('rel_improve_vs_topk', float('nan')):>10.6g} | "
            f"{m.get('loss_sae_full', float('nan')):>8.6g} | "
            f"{m.get('loss_sae_topk', float('nan')):>8.6g}"
        )

    print_row("scalar", scalar_metrics)
    print_row("residual_adv", residual_metrics)
    print_row("manifold", manifold_metrics)

    if manifold_metrics is not None:
        print("\nManifold delta diagnostics:")
        keys = [
            "delta_over_mu_mean",
            "delta_over_mu_median",
            "delta_over_mu_max",
            "weighted_delta_over_mu",
            "delta_norm_mean",
            "mu_norm_mean",
            "parallel_fraction_mean",
            "cos_delta_mu_mean",
            "offset_scale",
        ]
        for k in keys:
            if k in manifold_metrics:
                print(f"  {k:<28}: {manifold_metrics[k]:.6g}")

    print("\nInterpretation:")

    if args.random_control != "none":
        print(
            "  This was a RANDOM CONTROL run. The key question is whether the "
            "manifold model still improves strongly when real activation geometry is destroyed."
        )
        if manifold_metrics is not None:
            mvf = manifold_metrics.get("rel_improve_vs_full", float("nan"))
            mvt = manifold_metrics.get("rel_improve_vs_topk", float("nan"))
            print(f"  manifold.rel_improve_vs_full = {mvf:.6g}")
            print(f"  manifold.rel_improve_vs_topk = {mvt:.6g}")
            if mvf > 0.2:
                print(
                    "  WARNING: large improvement on random-control data. This suggests "
                    "capacity/overfitting may explain a significant part of the effect."
                )
            elif mvf > 0.05:
                print(
                    "  MODERATE WARNING: random-control improvement is nontrivial. "
                    "Compare against the real-activation run."
                )
            else:
                print(
                    "  GOOD SIGN: random-control improvement is small. This supports "
                    "the idea that real activation geometry matters."
                )

    else:
        if manifold_metrics is not None:
            print(
                f"  Manifold context mode = {args.context_mode}. "
                "Interpret this together with the other context modes."
            )
            if args.context_mode == "raw_x":
                print(
                    "  raw_x is a high-information/leaky control: strong performance here alone "
                    "does not establish feature-local manifold structure."
                )
            elif args.context_mode == "sae_recon":
                print(
                    "  sae_recon hides raw x from the offset MLP. Strong performance here is "
                    "better evidence for SAE-derived contextual structure."
                )
            elif args.context_mode == "ids_only":
                print(
                    "  ids_only is the strictest mode. Strong performance here suggests useful "
                    "correction can be inferred from active feature identity and coefficient alone."
                )

        if manifold_metrics is not None and scalar_metrics is not None:
            s = scalar_metrics.get("rel_improve_vs_full", float("nan"))
            m = manifold_metrics.get("rel_improve_vs_full", float("nan"))
            gap = m - s
            print(f"  manifold_vs_full - scalar_vs_full = {gap:.6g}")
            if gap > 0.1:
                print(
                    "  The manifold model beats scalar rescaling by a substantial margin. "
                    "This suggests the improvement is not merely amplitude correction."
                )
            else:
                print(
                    "  The manifold model does not clearly beat scalar rescaling. "
                    "This weakens the feature-manifold interpretation."
                )

        if manifold_metrics is not None and residual_metrics is not None:
            m = manifold_metrics.get("rel_improve_vs_full", float("nan"))
            r = residual_metrics.get("rel_improve_vs_full", float("nan"))
            gap = m - r
            print(f"  manifold_vs_full - residual_adversary_vs_full = {gap:.6g}")
            if gap > 0.05:
                print(
                    "  The manifold model beats the same-size generic residual adversary. "
                    "This supports feature-local structure over pure capacity."
                )
            elif gap < -0.05:
                print(
                    "  The same-size generic residual adversary beats the manifold model. "
                    "This suggests capacity may be the main driver."
                )
            else:
                print(
                    "  The manifold model and same-size residual adversary are similar. "
                    "This is ambiguous: capacity may explain much of the gain."
                )

        if manifold_metrics is not None:
            ratio = manifold_metrics.get("weighted_delta_over_mu", float("nan"))
            if ratio < 0.1:
                print(
                    "  weighted_delta_over_mu is small. This supports a local-correction "
                    "interpretation."
                )
            elif ratio < 0.5:
                print(
                    "  weighted_delta_over_mu is moderate. The manifold interpretation is "
                    "plausible, but the offsets are not tiny."
                )
            else:
                print(
                    "  weighted_delta_over_mu is large. The model may be replacing SAE "
                    "features rather than making small local manifold corrections."
                )

    print("\nRecommended next comparison:")
    print("  Run the same command with --random-control shuffle_dims.")
    print("  Then compare real manifold.rel_improve_vs_full against random-control manifold.rel_improve_vs_full.")
    print("=" * 90)


# ============================================================
# 11. Main experiment
# ============================================================


def run(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    device = args.device or get_device()
    print(f"Using device: {device}")

    tl_model, sae, cfg_dict = load_model_and_sae(
        model_name=args.model_name,
        release=args.release,
        sae_id=args.sae_id,
        device=device,
    )

    hook_name = infer_hook_name(sae, args.sae_id)
    print(f"Using hook_name: {hook_name}")

    d_sae, d_model = infer_dims(sae)
    print(f"SAE dimensions: d_sae={d_sae}, d_model={d_model}")

    train_loader, val_loader = make_activation_loaders(
        tl_model=tl_model,
        dataset=args.dataset,
        hook_name=hook_name,
        seq_len=args.seq_len,
        token_batch_size=args.token_batch_size,
        activation_batch_size=args.activation_batch_size,
        device=device,
        max_examples=args.max_examples,
        max_tokens=args.max_tokens,
        max_token_blocks=args.max_token_blocks,
        val_fraction=args.val_fraction,
        seed=args.seed,
        random_control=args.random_control,
    )

    scalar_metrics = None
    residual_metrics = None
    manifold_metrics = None

    scalar_params = None
    residual_params = None
    manifold_params = None

    # ------------------------------------------------------------
    # Scalar baseline
    # ------------------------------------------------------------

    if not args.skip_scalar:
        print("\n" + "=" * 80)
        print("Training scalar-rescaling baseline")
        print("=" * 80)

        scalar_model = ScalarRescaleSAE(
            sae=sae,
            d_model=d_model,
            top_k=args.top_k,
            hidden_dim=args.scalar_hidden_dim,
        )

        scalar_params = count_trainable_params(scalar_model)

        train_model(
            model=scalar_model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            device=device,
            log_every=args.log_every,
            save_path=args.scalar_save_path,
            model_name="scalar",
        )

        scalar_metrics = evaluate_model(
            scalar_model,
            val_loader,
            device=device,
        )

        print(f"[final scalar] {format_metrics(scalar_metrics)}")

        del scalar_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Manifold model
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Building manifold-refined SAE")
    print("=" * 80)

    manifold_cfg = ManifoldSAEConfig(
        d_model=d_model,
        d_sae=d_sae,
        top_k=args.top_k,
        rank=args.rank,
        hidden_dim=args.manifold_hidden_dim,
        lambda_delta=args.lambda_delta,
        lambda_z=args.lambda_z,
        lambda_parallel=args.lambda_parallel,
        normalize_decoder=args.normalize_decoder,
        use_offset_scale=not args.disable_offset_scale,
        offset_scale_init=args.offset_scale_init,
        context_mode=args.context_mode,
        sae_context_source=args.sae_context_source,
    )

    manifold_model = ManifoldRefinedSAE(
        sae=sae,
        cfg=manifold_cfg,
    )

    manifold_params = count_trainable_params(manifold_model)

    print_param_count("manifold", manifold_model)

    # ------------------------------------------------------------
    # Same-size residual adversary
    # ------------------------------------------------------------

    if not args.skip_residual_adversary:
        print("\n" + "=" * 80)
        print("Training same-size generic residual MLP adversary")
        print("=" * 80)

        residual_model = CapacityMatchedResidualMLP(
            sae=sae,
            d_model=d_model,
            top_k=args.top_k,
            target_params=manifold_params,
            n_hidden_layers=args.residual_hidden_layers,
            hidden_dim=args.residual_hidden_dim,
            residual_scale_init=args.residual_scale_init,
        )

        residual_params = count_trainable_params(residual_model)

        train_model(
            model=residual_model,
            train_loader=train_loader,
            val_loader=val_loader,
            lr=args.lr,
            weight_decay=args.weight_decay,
            num_epochs=args.num_epochs,
            device=device,
            log_every=args.log_every,
            save_path=args.residual_save_path,
            model_name="residual_adv",
        )

        residual_metrics = evaluate_model(
            residual_model,
            val_loader,
            device=device,
        )

        print(f"[final residual_adv] {format_metrics(residual_metrics)}")

        del residual_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Train manifold model
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("Training manifold-refined SAE")
    print("=" * 80)

    train_model(
        model=manifold_model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device=device,
        log_every=args.log_every,
        save_path=args.manifold_save_path,
        model_name="manifold",
    )

    manifold_metrics = evaluate_model(
        manifold_model,
        val_loader,
        device=device,
    )

    print(f"[final manifold] {format_metrics(manifold_metrics)}")

    summarize_experiment(
        args=args,
        scalar_metrics=scalar_metrics,
        residual_metrics=residual_metrics,
        manifold_metrics=manifold_metrics,
        train_batches=len(train_loader),
        val_batches=len(val_loader),
        scalar_params=scalar_params,
        residual_params=residual_params,
        manifold_params=manifold_params,
    )


# ============================================================
# 12. CLI
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # SAE / model
    p.add_argument("--model-name", type=str, default="gpt2-small")
    p.add_argument("--release", type=str, default="gpt2-small-res-jb")
    p.add_argument("--sae-id", type=str, default="blocks.6.hook_resid_pre")

    # Data
    p.add_argument(
        "--dataset",
        type=str,
        default="tinystories",
        choices=["tinyshakespeare", "tinystories"],
    )
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--max-examples", type=int, default=20000)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    p.add_argument("--max-token-blocks", type=int, default=8192)
    p.add_argument("--val-fraction", type=float, default=0.05)

    # Random controls
    p.add_argument(
        "--random-control",
        type=str,
        default="none",
        choices=["none", "gaussian", "shuffle_dims", "permute_examples"],
    )

    # Batch sizes
    p.add_argument("--token-batch-size", type=int, default=8)
    p.add_argument("--activation-batch-size", type=int, default=512)

    # Architecture
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--manifold-hidden-dim", type=int, default=256)
    p.add_argument("--scalar-hidden-dim", type=int, default=128)
    p.add_argument(
        "--context-mode",
        type=str,
        default="sae_recon",
        choices=["ids_only", "sae_recon", "raw_x"],
        help=(
            "What information the manifold offset MLP can see: "
            "ids_only = f_i and v_i only; "
            "sae_recon = f_i, v_i, and SAE reconstruction context; "
            "raw_x = f_i, v_i, and original activation x."
        ),
    )
    p.add_argument(
        "--sae-context-source",
        type=str,
        default="topk",
        choices=["topk", "full"],
        help=(
            "For --context-mode sae_recon, choose x_hat_sae_topk or "
            "x_hat_sae_full as the context. Use full to reproduce the original script."
        ),
    )

    # Manifold loss
    p.add_argument("--lambda-delta", type=float, default=1e-2)
    p.add_argument("--lambda-z", type=float, default=1e-3)
    p.add_argument("--lambda-parallel", type=float, default=1e-2)
    p.add_argument("--normalize-decoder", action="store_true")

    # Offset scaling
    p.add_argument("--disable-offset-scale", action="store_true")
    p.add_argument("--offset-scale-init", type=float, default=-5.0)

    # Residual adversary
    p.add_argument("--skip-residual-adversary", action="store_true")
    p.add_argument("--residual-hidden-layers", type=int, default=1)
    p.add_argument("--residual-hidden-dim", type=int, default=None)
    p.add_argument("--residual-scale-init", type=float, default=-5.0)

    # Scalar
    p.add_argument("--skip-scalar", action="store_true")

    # Optimization
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--log-every", type=int, default=100)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scalar-save-path", type=str, default="checkpoints/scalar_rescale.pt")
    p.add_argument("--residual-save-path", type=str, default="checkpoints/residual_adversary.pt")
    p.add_argument("--manifold-save-path", type=str, default="checkpoints/manifold_refined.pt")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)