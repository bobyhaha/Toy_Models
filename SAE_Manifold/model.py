"""
SAE Lens Manifold-Refined SAE Experiment
=======================================

This script:
1. Loads GPT-2 small with TransformerLens.
2. Loads a pretrained SAE from SAE Lens.
3. Builds activation datasets from TinyShakespeare and/or TinyStories.
4. Freezes the SAE.
5. Trains:
   - ManifoldRefinedSAE: feature-conditioned local decoder offsets.
   - ScalarRescaleSAE: scalar-amplitude baseline.
6. Evaluates reconstruction improvement against the frozen SAE.

Install:
    pip install torch transformer-lens sae-lens datasets tqdm

Example:
    python sae_lens_manifold_experiment.py \
        --dataset tinyshakespeare \
        --release gpt2-small-res-jb \
        --sae-id blocks.6.hook_resid_pre \
        --num-epochs 3 \
        --max-token-blocks 256

Notes:
- The SAE is pretrained and frozen.
- The activations are taken from sae.cfg.hook_name, e.g. blocks.6.hook_resid_pre.
- For this particular SAE family, d_sae is usually 24576 and d_model is 768.
"""

import argparse
import math
import os
import urllib.request
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

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


# ============================================================
# 1. Text loading
# ============================================================


def load_tinyshakespeare_text() -> str:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    with urllib.request.urlopen(url) as f:
        return f.read().decode("utf-8")


def load_tinystories_texts(split: str = "train", max_examples: Optional[int] = None) -> List[str]:
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
    Stores fixed-length token blocks produced by TransformerLens tokenizer.
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
        total = 0

        for text in tqdm(texts, desc="Tokenizing"):
            # prepend_bos=False keeps the dataset close to ordinary raw-token blocks.
            toks = model.to_tokens(text, prepend_bos=False).squeeze(0).cpu()
            token_chunks.append(toks)
            total += toks.numel()
            if max_tokens is not None and total >= max_tokens:
                break

        all_tokens = torch.cat(token_chunks, dim=0)
        if max_tokens is not None:
            all_tokens = all_tokens[:max_tokens]

        n_blocks = all_tokens.numel() // seq_len
        all_tokens = all_tokens[: n_blocks * seq_len]
        self.tokens = all_tokens.view(n_blocks, seq_len).long()

        if len(self.tokens) == 0:
            raise ValueError("No token blocks were created. Increase max_tokens or reduce seq_len.")

    def __len__(self) -> int:
        return self.tokens.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.tokens[idx]


# ============================================================
# 2. Activation dataset through TransformerLens cache
# ============================================================


class ActivationDataset(Dataset):
    """
    Materializes activations from one hook point into memory.

    For larger experiments, replace this with a streaming buffer.
    For a first experiment, materializing is simpler and safer.
    """

    def __init__(
        self,
        tl_model: HookedTransformer,
        token_dataset: TokenBlockDataset,
        hook_name: str,
        token_batch_size: int,
        device: str,
        max_token_blocks: Optional[int] = None,
        token_position: str = "all",
    ):
        self.activations: torch.Tensor
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
                _, cache = tl_model.run_with_cache(tokens, names_filter=[hook_name])
                acts = cache[hook_name]

                # Expected shape: [batch, seq_len, d_model]
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

        self.activations = torch.cat(acts_cpu, dim=0)

    def __len__(self) -> int:
        return self.activations.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
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

    n_val = max(1, int(len(act_dataset) * val_fraction))
    n_train = len(act_dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(act_dataset, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_set,
        batch_size=activation_batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=activation_batch_size,
        shuffle=False,
        drop_last=False,
    )

    return train_loader, val_loader


# ============================================================
# 3. SAE wrapper
# ============================================================


class FrozenSAEWrapper(nn.Module):
    """
    Small compatibility wrapper.

    The rest of this script expects:
        encode(x) -> sparse feature activations
        decode(f) -> reconstruction
        W_dec: [d_sae, d_model]
        b_dec: [d_model]

    SAE Lens exposes these for standard pretrained SAEs.
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
        # Most SAE Lens SAEs expose b_dec. This fallback makes failures clearer.
        if not hasattr(self.sae, "b_dec"):
            raise AttributeError("This SAE object does not expose b_dec.")
        return self.sae.b_dec

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        if hasattr(self.sae, "decode"):
            return self.sae.decode(f)
        return self.b_dec + f @ self.W_dec


# ============================================================
# 4. Manifold-refined SAE
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


class OffsetMLP(nn.Module):
    """
    Predicts low-dimensional coordinates z_i(x) for each active feature.

    Per active feature input:
        [feature_activation, decoder_vector, sae_reconstruction]
    Output:
        z_i in R^rank
    """

    def __init__(self, d_model: int, rank: int, hidden_dim: int):
        super().__init__()
        in_dim = 1 + d_model + d_model
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, rank),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        f_active: torch.Tensor,
        v_active: torch.Tensor,
        x_hat_sae: torch.Tensor,
    ) -> torch.Tensor:
        batch, k, d_model = v_active.shape
        x_context = x_hat_sae[:, None, :].expand(batch, k, d_model)
        f_context = f_active[..., None]
        inp = torch.cat([f_context, v_active, x_context], dim=-1)
        return self.net(inp)


class ManifoldRefinedSAE(nn.Module):
    """
    Frozen SAE plus feature-conditioned low-rank local offsets.

    SAE:
        x_hat = b + sum_i f_i(x) v_i

    Refined top-k model:
        x_hat = b + sum_{i in top-k} f_i(x) [v_i + U_i z_i(x)]
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
        )

        self.U = nn.Parameter(0.01 * torch.randn(cfg.d_sae, cfg.d_model, cfg.rank))

    def get_decoder_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sae.W_dec, self.sae.b_dec

    @torch.no_grad()
    def encode_frozen(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        W_dec, b_dec = self.get_decoder_params()
        W_use = F.normalize(W_dec, dim=-1) if cfg.normalize_decoder else W_dec

        with torch.no_grad():
            f = self.encode_frozen(x)
            x_hat_sae_full = b_dec + f @ W_use
            f_active, active_idx = torch.topk(
                f,
                k=min(cfg.top_k, f.shape[-1]),
                dim=-1,
                largest=True,
                sorted=False,
            )

        v_active = W_use[active_idx]
        z = self.offset_mlp(f_active=f_active, v_active=v_active, x_hat_sae=x_hat_sae_full)
        U_active = self.U[active_idx]
        delta = torch.einsum("bkdr,bkr->bkd", U_active, z)

        refined_atoms = v_active + delta
        x_hat_refined = b_dec + (f_active[..., None] * refined_atoms).sum(dim=1)
        x_hat_sae_topk = b_dec + (f_active[..., None] * v_active).sum(dim=1)

        return {
            "x_hat_refined": x_hat_refined,
            "x_hat_sae_full": x_hat_sae_full,
            "x_hat_sae_topk": x_hat_sae_topk,
            "f": f,
            "f_active": f_active,
            "active_idx": active_idx,
            "v_active": v_active,
            "z": z,
            "delta": delta,
        }

    def compute_loss(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        out = self.forward(x)
        cfg = self.cfg

        x_hat_refined = out["x_hat_refined"]
        x_hat_sae_full = out["x_hat_sae_full"]
        x_hat_sae_topk = out["x_hat_sae_topk"]
        f_active = out["f_active"]
        v_active = out["v_active"]
        delta = out["delta"]
        z = out["z"]

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
            residual_r2 = 1.0 - residual_after / residual_before
            relative_improvement_vs_topk = (loss_sae_topk - loss_rec) / loss_sae_topk.clamp_min(1e-8)
            relative_improvement_vs_full = (loss_sae_full - loss_rec) / loss_sae_full.clamp_min(1e-8)
            mean_l0 = (out["f"] > 0).float().sum(dim=-1).mean()

        logs = {
            "loss": loss.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_sae_full": loss_sae_full.detach(),
            "loss_sae_topk": loss_sae_topk.detach(),
            "loss_delta": loss_delta.detach(),
            "loss_z": loss_z.detach(),
            "loss_parallel": loss_parallel.detach(),
            "residual_r2_vs_topk": residual_r2.detach(),
            "rel_improve_vs_topk": relative_improvement_vs_topk.detach(),
            "rel_improve_vs_full": relative_improvement_vs_full.detach(),
            "mean_l0": mean_l0.detach(),
        }
        return loss, logs


# ============================================================
# 5. Scalar-rescaling baseline
# ============================================================


class ScalarRescaleSAE(nn.Module):
    """
    Baseline:
        x_hat = b + sum_i alpha_i(x) f_i(x) v_i

    This checks whether your offset model is only changing amplitudes.
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
        nn.init.zeros_(self.alpha_mlp[-1].weight)
        nn.init.zeros_(self.alpha_mlp[-1].bias)

    @torch.no_grad()
    def encode_frozen(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        W_dec, b_dec = self.sae.W_dec, self.sae.b_dec

        with torch.no_grad():
            f = self.encode_frozen(x)
            x_hat_sae_full = b_dec + f @ W_dec
            f_active, active_idx = torch.topk(
                f,
                k=min(self.top_k, f.shape[-1]),
                dim=-1,
                largest=True,
                sorted=False,
            )

        v_active = W_dec[active_idx]
        batch, k, d_model = v_active.shape
        x_context = x_hat_sae_full[:, None, :].expand(batch, k, d_model)
        inp = torch.cat([f_active[..., None], v_active, x_context], dim=-1)

        # Near-identity initialization.
        alpha = 1.0 + 0.1 * self.alpha_mlp(inp).squeeze(-1)
        x_hat_rescaled = b_dec + (alpha[..., None] * f_active[..., None] * v_active).sum(dim=1)
        x_hat_sae_topk = b_dec + (f_active[..., None] * v_active).sum(dim=1)

        return {
            "x_hat_rescaled": x_hat_rescaled,
            "x_hat_sae_full": x_hat_sae_full,
            "x_hat_sae_topk": x_hat_sae_topk,
            "alpha": alpha,
            "active_idx": active_idx,
            "f_active": f_active,
            "f": f,
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
            relative_improvement_vs_topk = (loss_sae_topk - loss) / loss_sae_topk.clamp_min(1e-8)
            relative_improvement_vs_full = (loss_sae_full - loss) / loss_sae_full.clamp_min(1e-8)
            mean_l0 = (out["f"] > 0).float().sum(dim=-1).mean()

        logs = {
            "loss": loss.detach(),
            "loss_sae_full": loss_sae_full.detach(),
            "loss_sae_topk": loss_sae_topk.detach(),
            "rel_improve_vs_topk": relative_improvement_vs_topk.detach(),
            "rel_improve_vs_full": relative_improvement_vs_full.detach(),
            "mean_l0": mean_l0.detach(),
        }
        return loss, logs


# ============================================================
# 6. Train/eval loops
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
    raise ValueError("Unknown batch format")


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    n = 0

    for batch in loader:
        x = move_batch_to_device(batch, device).float()
        _, logs = model.compute_loss(x)
        for k, v in logs.items():
            val = v.item() if torch.is_tensor(v) else float(v)
            totals[k] = totals.get(k, 0.0) + val
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def format_metrics(metrics: Dict[str, float]) -> str:
    return " ".join(f"{k}={v:.6g}" for k, v in metrics.items())


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
) -> nn.Module:
    model.to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    step = 0
    best_val = math.inf

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            x = move_batch_to_device(batch, device).float()
            loss, logs = model.compute_loss(x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % log_every == 0:
                print(f"epoch={epoch} step={step} {format_metrics({k: v.item() for k, v in logs.items()})}")
            step += 1

        if val_loader is not None:
            metrics = evaluate_model(model, val_loader, device=device)
            print(f"[val epoch={epoch}] {format_metrics(metrics)}")

            val_loss = metrics.get("loss", math.inf)
            if save_path is not None and val_loss < best_val:
                best_val = val_loss
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model to {save_path}")

    return model


# ============================================================
# 7. Loading SAE Lens SAE + running experiment
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

    # Some old/example code calls fold_W_dec_norm(). It is useful for dashboards,
    # but for reconstruction experiments it can change feature scales. Leave off
    # unless you know you want the folded representation.

    wrapped = FrozenSAEWrapper(sae)
    return tl_model, wrapped, cfg_dict


def infer_dims(sae: FrozenSAEWrapper) -> Tuple[int, int]:
    W_dec = sae.W_dec
    if W_dec.ndim != 2:
        raise ValueError(f"Expected W_dec to be 2D, got shape {tuple(W_dec.shape)}")
    d_sae, d_model = W_dec.shape
    return d_sae, d_model


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

    hook_name = getattr(sae.sae.cfg, "hook_name", None)
    if hook_name is None:
        hook_name = args.sae_id
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
    )

    print("\nTraining scalar-rescaling baseline...")
    scalar_model = ScalarRescaleSAE(
        sae=sae,
        d_model=d_model,
        top_k=args.top_k,
        hidden_dim=args.scalar_hidden_dim,
    )
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
    )
    scalar_metrics = evaluate_model(scalar_model, val_loader, device=device)
    print(f"[final scalar] {format_metrics(scalar_metrics)}")

    print("\nTraining manifold-refined SAE...")
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
    )
    manifold_model = ManifoldRefinedSAE(sae=sae, cfg=manifold_cfg)
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
    )
    manifold_metrics = evaluate_model(manifold_model, val_loader, device=device)
    print(f"[final manifold] {format_metrics(manifold_metrics)}")

    print("\nSummary:")
    print(f"scalar.rel_improve_vs_full   = {scalar_metrics.get('rel_improve_vs_full', float('nan')):.6g}")
    print(f"manifold.rel_improve_vs_full = {manifold_metrics.get('rel_improve_vs_full', float('nan')):.6g}")
    print(f"scalar.rel_improve_vs_topk   = {scalar_metrics.get('rel_improve_vs_topk', float('nan')):.6g}")
    print(f"manifold.rel_improve_vs_topk = {manifold_metrics.get('rel_improve_vs_topk', float('nan')):.6g}")


# ============================================================
# 8. CLI
# ============================================================


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # SAE / model
    p.add_argument("--model-name", type=str, default="gpt2-small")
    p.add_argument("--release", type=str, default="gpt2-small-res-jb")
    p.add_argument("--sae-id", type=str, default="blocks.6.hook_resid_pre")

    # Data
    p.add_argument("--dataset", type=str, default="tinyshakespeare", choices=["tinyshakespeare", "tinystories"])
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=2_000_000)
    p.add_argument("--max-token-blocks", type=int, default=256)
    p.add_argument("--val-fraction", type=float, default=0.05)

    # Batches
    p.add_argument("--token-batch-size", type=int, default=8)
    p.add_argument("--activation-batch-size", type=int, default=1024)

    # Model hyperparams
    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--manifold-hidden-dim", type=int, default=256)
    p.add_argument("--scalar-hidden-dim", type=int, default=128)
    p.add_argument("--lambda-delta", type=float, default=1e-3)
    p.add_argument("--lambda-z", type=float, default=1e-4)
    p.add_argument("--lambda-parallel", type=float, default=1e-3)
    p.add_argument("--normalize-decoder", action="store_true")

    # Optim
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-epochs", type=int, default=3)
    p.add_argument("--log-every", type=int, default=50)

    # Misc
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scalar-save-path", type=str, default="checkpoints/scalar_rescale.pt")
    p.add_argument("--manifold-save-path", type=str, default="checkpoints/manifold_refined.pt")

    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
