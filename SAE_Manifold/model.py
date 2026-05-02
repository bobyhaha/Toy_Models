import math
import urllib.request
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


# ============================================================
# 1. Dataset utilities
# ============================================================

def load_tinyshakespeare_text() -> str:
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    text = urllib.request.urlopen(url).read().decode("utf-8")
    return text


def load_tinystories_texts(split: str = "train", max_examples: Optional[int] = None) -> List[str]:
    ds = load_dataset("roneneldan/TinyStories", split=split)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    return [ex["text"] for ex in ds]


class TokenDataset(Dataset):
    """
    Converts raw text into fixed-length token blocks.
    """

    def __init__(
        self,
        tokenizer,
        texts: List[str],
        seq_len: int = 128,
        max_tokens: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        all_ids = []

        for text in tqdm(texts, desc="Tokenizing texts"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)

            if max_tokens is not None and len(all_ids) >= max_tokens:
                all_ids = all_ids[:max_tokens]
                break

        n_blocks = len(all_ids) // seq_len
        all_ids = all_ids[: n_blocks * seq_len]

        self.tokens = torch.tensor(all_ids, dtype=torch.long).view(n_blocks, seq_len)

    def __len__(self):
        return self.tokens.shape[0]

    def __getitem__(self, idx):
        return self.tokens[idx]


def make_text_dataloader(
    dataset_name: str,
    tokenizer,
    split: str = "train",
    seq_len: int = 128,
    batch_size: int = 16,
    max_examples: Optional[int] = None,
    max_tokens: Optional[int] = None,
    shuffle: bool = True,
):
    if dataset_name.lower() in ["tinyshakespeare", "shakespeare"]:
        text = load_tinyshakespeare_text()
        texts = [text]

    elif dataset_name.lower() in ["tinystories", "tiny_stories"]:
        texts = load_tinystories_texts(split=split, max_examples=max_examples)

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    dataset = TokenDataset(
        tokenizer=tokenizer,
        texts=texts,
        seq_len=seq_len,
        max_tokens=max_tokens,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return loader


# ============================================================
# 2. Activation extraction
# ============================================================

class ActivationBuffer:
    """
    Collects activations from a specified transformer block.

    This version is written for GPT-like HuggingFace models where the blocks live at:
        model.transformer.h[layer_idx]

    For other models, modify get_layer().
    """

    def __init__(
        self,
        model: nn.Module,
        layer_idx: int,
        device: str = "cuda",
        token_position: str = "all",
    ):
        self.model = model
        self.layer_idx = layer_idx
        self.device = device
        self.token_position = token_position
        self.cached_activation = None
        self.hook_handle = None

    def get_layer(self):
        # GPT-2 style.
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h[self.layer_idx]

        # Pythia / GPT-NeoX style.
        if hasattr(self.model, "gpt_neox") and hasattr(self.model.gpt_neox, "layers"):
            return self.model.gpt_neox.layers[self.layer_idx]

        # LLaMA style.
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers[self.layer_idx]

        raise ValueError("Unknown model architecture. Modify ActivationBuffer.get_layer().")

    def hook_fn(self, module, inputs, output):
        # Many HF blocks return either tensor or tuple.
        if isinstance(output, tuple):
            act = output[0]
        else:
            act = output

        # act: [batch, seq_len, d_model]
        self.cached_activation = act.detach()

    def __enter__(self):
        layer = self.get_layer()
        self.hook_handle = layer.register_forward_hook(self.hook_fn)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.hook_handle is not None:
            self.hook_handle.remove()

    @torch.no_grad()
    def get_activations(self, input_ids: torch.Tensor):
        self.cached_activation = None

        input_ids = input_ids.to(self.device)

        _ = self.model(input_ids)

        acts = self.cached_activation

        if acts is None:
            raise RuntimeError("Hook failed to capture activations.")

        # acts: [batch, seq_len, d_model]
        if self.token_position == "all":
            acts = acts.reshape(-1, acts.shape[-1])

        elif self.token_position == "last":
            acts = acts[:, -1, :]

        elif isinstance(self.token_position, int):
            acts = acts[:, self.token_position, :]

        else:
            raise ValueError(f"Unknown token_position: {self.token_position}")

        return acts


class ActivationDataset(Dataset):
    """
    Materializes activations into memory.

    For large experiments, you should stream activations instead.
    This is for a first pilot.
    """

    def __init__(
        self,
        model,
        token_loader,
        layer_idx: int,
        device: str = "cuda",
        token_position: str = "all",
        max_activation_batches: Optional[int] = None,
    ):
        self.activations = []

        model.eval()
        model.to(device)

        with ActivationBuffer(
            model=model,
            layer_idx=layer_idx,
            device=device,
            token_position=token_position,
        ) as buffer:
            for batch_idx, input_ids in enumerate(tqdm(token_loader, desc="Collecting activations")):
                if max_activation_batches is not None and batch_idx >= max_activation_batches:
                    break

                acts = buffer.get_activations(input_ids)
                self.activations.append(acts.cpu())

        self.activations = torch.cat(self.activations, dim=0)

    def __len__(self):
        return self.activations.shape[0]

    def __getitem__(self, idx):
        return self.activations[idx]


def make_activation_loader(
    model,
    token_loader,
    layer_idx: int,
    activation_batch_size: int = 1024,
    device: str = "cuda",
    token_position: str = "all",
    max_activation_batches: Optional[int] = None,
    shuffle: bool = True,
):
    dataset = ActivationDataset(
        model=model,
        token_loader=token_loader,
        layer_idx=layer_idx,
        device=device,
        token_position=token_position,
        max_activation_batches=max_activation_batches,
    )

    loader = DataLoader(
        dataset,
        batch_size=activation_batch_size,
        shuffle=shuffle,
        drop_last=True,
    )

    return loader


# ============================================================
# 3. Manifold-refined SAE
# ============================================================

@dataclass
class ManifoldSAEConfig:
    d_model: int
    n_features: int
    top_k: int = 32
    rank: int = 4
    hidden_dim: int = 256

    lambda_delta: float = 1e-3
    lambda_z: float = 1e-4
    lambda_parallel: float = 1e-3

    normalize_decoder: bool = False


class OffsetMLP(nn.Module):
    """
    Predicts z_i(x), the low-dimensional local coordinate
    for active feature i.

    Input per active feature:
        [f_i, v_i, x_hat_sae]

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

        # Make initial correction close to zero.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        f_active: torch.Tensor,
        v_active: torch.Tensor,
        x_hat_sae: torch.Tensor,
    ) -> torch.Tensor:
        """
        f_active: [batch, k]
        v_active: [batch, k, d_model]
        x_hat_sae: [batch, d_model]

        returns:
            z: [batch, k, rank]
        """

        batch, k, d_model = v_active.shape

        x_context = x_hat_sae[:, None, :].expand(batch, k, d_model)
        f_context = f_active[..., None]

        inp = torch.cat([f_context, v_active, x_context], dim=-1)

        return self.net(inp)


class ManifoldRefinedSAE(nn.Module):
    """
    Frozen SAE plus feature-conditioned low-rank manifold offsets.

    Standard SAE:
        x_hat_sae = b + sum_i f_i(x) v_i

    Manifold-refined SAE:
        x_hat = b + sum_{i active} f_i(x) (v_i + U_i z_i(x))
    """

    def __init__(self, sae: nn.Module, cfg: ManifoldSAEConfig):
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

        self.U = nn.Parameter(
            0.01 * torch.randn(cfg.n_features, cfg.d_model, cfg.rank)
        )

    def get_decoder_params(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Modify this if your SAE names are different.
        Expected:
            W_dec: [n_features, d_model]
            b_dec: [d_model]
        """

        W_dec = self.sae.W_dec
        b_dec = self.sae.b_dec

        return W_dec, b_dec

    @torch.no_grad()
    def encode_frozen(self, x: torch.Tensor) -> torch.Tensor:
        return self.sae.encode(x)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cfg = self.cfg

        W_dec, b_dec = self.get_decoder_params()

        if cfg.normalize_decoder:
            W_use = F.normalize(W_dec, dim=-1)
        else:
            W_use = W_dec

        with torch.no_grad():
            f = self.encode_frozen(x)

            x_hat_sae_full = b_dec + f @ W_use

            f_active, active_idx = torch.topk(
                f,
                k=cfg.top_k,
                dim=-1,
                largest=True,
                sorted=False,
            )

        v_active = W_use[active_idx]

        z = self.offset_mlp(
            f_active=f_active,
            v_active=v_active,
            x_hat_sae=x_hat_sae_full,
        )

        U_active = self.U[active_idx]

        delta = torch.einsum("bkdr,bkr->bkd", U_active, z)

        refined_atoms = v_active + delta
        contrib = f_active[..., None] * refined_atoms

        x_hat_refined = b_dec + contrib.sum(dim=1)

        # For fair comparison, also compute top-k-only SAE reconstruction.
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

            r_sae = x - x_hat_sae_topk
            r_offset = x_hat_refined - x_hat_sae_topk

            residual_mse_after = (r_sae - r_offset).pow(2).mean()
            residual_mse_before = r_sae.pow(2).mean().clamp_min(1e-8)

            residual_r2 = 1.0 - residual_mse_after / residual_mse_before

            relative_improvement_vs_topk = (
                loss_sae_topk - loss_rec
            ) / loss_sae_topk.clamp_min(1e-8)

            relative_improvement_vs_full = (
                loss_sae_full - loss_rec
            ) / loss_sae_full.clamp_min(1e-8)

        logs = {
            "loss": loss.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_sae_full": loss_sae_full.detach(),
            "loss_sae_topk": loss_sae_topk.detach(),
            "loss_delta": loss_delta.detach(),
            "loss_z": loss_z.detach(),
            "loss_parallel": loss_parallel.detach(),
            "residual_r2": residual_r2.detach(),
            "relative_improvement_vs_topk": relative_improvement_vs_topk.detach(),
            "relative_improvement_vs_full": relative_improvement_vs_full.detach(),
        }

        return loss, logs


# ============================================================
# 4. Scalar-rescaling baseline
# ============================================================

class ScalarRescaleSAE(nn.Module):
    """
    Baseline:
        x_hat = b + sum_i alpha_i(x) f_i(x) v_i

    This tests whether the manifold offset is doing more than changing feature amplitude.
    """

    def __init__(
        self,
        sae: nn.Module,
        d_model: int,
        n_features: int,
        top_k: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.sae = sae
        self.d_model = d_model
        self.n_features = n_features
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

    def get_decoder_params(self):
        W_dec = self.sae.W_dec
        b_dec = self.sae.b_dec
        return W_dec, b_dec

    @torch.no_grad()
    def encode_frozen(self, x):
        return self.sae.encode(x)

    def forward(self, x):
        W_dec, b_dec = self.get_decoder_params()

        with torch.no_grad():
            f = self.encode_frozen(x)
            x_hat_sae_full = b_dec + f @ W_dec

            f_active, active_idx = torch.topk(
                f,
                k=self.top_k,
                dim=-1,
                largest=True,
                sorted=False,
            )

        v_active = W_dec[active_idx]

        batch, k, d_model = v_active.shape

        x_context = x_hat_sae_full[:, None, :].expand(batch, k, d_model)
        f_context = f_active[..., None]

        inp = torch.cat([f_context, v_active, x_context], dim=-1)

        alpha = 1.0 + 0.1 * self.alpha_mlp(inp).squeeze(-1)

        x_hat_rescaled = b_dec + (
            alpha[..., None] * f_active[..., None] * v_active
        ).sum(dim=1)

        x_hat_sae_topk = b_dec + (f_active[..., None] * v_active).sum(dim=1)

        return {
            "x_hat_rescaled": x_hat_rescaled,
            "x_hat_sae_full": x_hat_sae_full,
            "x_hat_sae_topk": x_hat_sae_topk,
            "alpha": alpha,
            "active_idx": active_idx,
            "f_active": f_active,
        }

    def compute_loss(self, x):
        out = self.forward(x)

        x_hat_rescaled = out["x_hat_rescaled"]
        x_hat_sae_full = out["x_hat_sae_full"]
        x_hat_sae_topk = out["x_hat_sae_topk"]

        loss = F.mse_loss(x_hat_rescaled, x)

        with torch.no_grad():
            loss_sae_full = F.mse_loss(x_hat_sae_full, x)
            loss_sae_topk = F.mse_loss(x_hat_sae_topk, x)

            relative_improvement_vs_topk = (
                loss_sae_topk - loss
            ) / loss_sae_topk.clamp_min(1e-8)

            relative_improvement_vs_full = (
                loss_sae_full - loss
            ) / loss_sae_full.clamp_min(1e-8)

        logs = {
            "loss": loss.detach(),
            "loss_sae_full": loss_sae_full.detach(),
            "loss_sae_topk": loss_sae_topk.detach(),
            "relative_improvement_vs_topk": relative_improvement_vs_topk.detach(),
            "relative_improvement_vs_full": relative_improvement_vs_full.detach(),
        }

        return loss, logs


# ============================================================
# 5. Training / evaluation loops
# ============================================================

def move_batch_to_device(batch, device):
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


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loaders: Optional[Dict[str, DataLoader]] = None,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 5,
    device: str = "cuda",
    log_every: int = 100,
):
    model.to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    step = 0

    for epoch in range(num_epochs):
        model.train()

        for batch in train_loader:
            x = move_batch_to_device(batch, device)

            loss, logs = model.compute_loss(x)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % log_every == 0:
                log_str = " ".join(
                    f"{k}={v.item():.6f}"
                    for k, v in logs.items()
                    if torch.is_tensor(v) and v.ndim == 0
                )
                print(f"epoch={epoch} step={step} {log_str}")

            step += 1

        if val_loaders is not None:
            for name, loader in val_loaders.items():
                metrics = evaluate_model(model, loader, device=device)
                metric_str = " ".join(f"{k}={v:.6f}" for k, v in metrics.items())
                print(f"[val:{name}] {metric_str}")


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cuda",
):
    model.eval()

    totals = {}
    n = 0

    for batch in loader:
        x = move_batch_to_device(batch, device)

        _, logs = model.compute_loss(x)

        for k, v in logs.items():
            if torch.is_tensor(v):
                totals[k] = totals.get(k, 0.0) + v.item()
            else:
                totals[k] = totals.get(k, 0.0) + float(v)

        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


# ============================================================
# 6. Cross-dataset experiment runner
# ============================================================

def build_activation_loaders_for_dataset(
    dataset_name: str,
    lm_model,
    tokenizer,
    layer_idx: int,
    seq_len: int,
    token_batch_size: int,
    activation_batch_size: int,
    max_examples: Optional[int],
    max_tokens: Optional[int],
    max_activation_batches: Optional[int],
    device: str,
):
    token_train_loader = make_text_dataloader(
        dataset_name=dataset_name,
        tokenizer=tokenizer,
        split="train",
        seq_len=seq_len,
        batch_size=token_batch_size,
        max_examples=max_examples,
        max_tokens=max_tokens,
        shuffle=True,
    )

    act_train_loader = make_activation_loader(
        model=lm_model,
        token_loader=token_train_loader,
        layer_idx=layer_idx,
        activation_batch_size=activation_batch_size,
        device=device,
        token_position="all",
        max_activation_batches=max_activation_batches,
        shuffle=True,
    )

    return act_train_loader


def run_cross_dataset_experiment(
    sae,
    lm_model_name: str = "gpt2",
    layer_idx: int = 6,
    d_model: int = 768,
    n_features: int = 16384,
    seq_len: int = 128,
    token_batch_size: int = 8,
    activation_batch_size: int = 1024,
    max_examples_tinystories: int = 5000,
    max_tokens_shakespeare: int = 2_000_000,
    max_activation_batches: int = 100,
    top_k: int = 32,
    rank: int = 4,
    device: str = "cuda",
):
    tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
    lm_model = AutoModelForCausalLM.from_pretrained(lm_model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lm_model.to(device)
    lm_model.eval()

    print("Building TinyShakespeare activations...")
    shakespeare_loader = build_activation_loaders_for_dataset(
        dataset_name="tinyshakespeare",
        lm_model=lm_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        seq_len=seq_len,
        token_batch_size=token_batch_size,
        activation_batch_size=activation_batch_size,
        max_examples=None,
        max_tokens=max_tokens_shakespeare,
        max_activation_batches=max_activation_batches,
        device=device,
    )

    print("Building TinyStories activations...")
    stories_loader = build_activation_loaders_for_dataset(
        dataset_name="tinystories",
        lm_model=lm_model,
        tokenizer=tokenizer,
        layer_idx=layer_idx,
        seq_len=seq_len,
        token_batch_size=token_batch_size,
        activation_batch_size=activation_batch_size,
        max_examples=max_examples_tinystories,
        max_tokens=None,
        max_activation_batches=max_activation_batches,
        device=device,
    )

    # ------------------------------------------------------------
    # Train on TinyShakespeare, evaluate on both
    # ------------------------------------------------------------

    cfg = ManifoldSAEConfig(
        d_model=d_model,
        n_features=n_features,
        top_k=top_k,
        rank=rank,
        hidden_dim=256,
        lambda_delta=1e-3,
        lambda_z=1e-4,
        lambda_parallel=1e-3,
        normalize_decoder=False,
    )

    print("\nTraining manifold model on TinyShakespeare...")
    manifold_shakespeare = ManifoldRefinedSAE(sae=sae, cfg=cfg)

    train_model(
        model=manifold_shakespeare,
        train_loader=shakespeare_loader,
        val_loaders={
            "tinyshakespeare": shakespeare_loader,
            "tinystories": stories_loader,
        },
        lr=1e-3,
        weight_decay=1e-4,
        num_epochs=5,
        device=device,
        log_every=100,
    )

    # ------------------------------------------------------------
    # Train on TinyStories, evaluate on both
    # ------------------------------------------------------------

    print("\nTraining manifold model on TinyStories...")
    manifold_stories = ManifoldRefinedSAE(sae=sae, cfg=cfg)

    train_model(
        model=manifold_stories,
        train_loader=stories_loader,
        val_loaders={
            "tinyshakespeare": shakespeare_loader,
            "tinystories": stories_loader,
        },
        lr=1e-3,
        weight_decay=1e-4,
        num_epochs=5,
        device=device,
        log_every=100,
    )

    return {
        "manifold_shakespeare": manifold_shakespeare,
        "manifold_stories": manifold_stories,
        "shakespeare_loader": shakespeare_loader,
        "stories_loader": stories_loader,
    }


# ============================================================
# 7. Example usage
# ============================================================

if __name__ == "__main__":
    """
    You need to load your trained SAE here.

    Example placeholder:

        sae = torch.load("my_sae.pt")

    Your SAE must expose:
        sae.encode(x)
        sae.W_dec
        sae.b_dec
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    raise NotImplementedError(
        "Load your trained SAE here, then call run_cross_dataset_experiment(sae=sae, ...)."
    )

    # Example:
    #
    # sae = torch.load("my_sae.pt", map_location=device)
    #
    # results = run_cross_dataset_experiment(
    #     sae=sae,
    #     lm_model_name="gpt2",
    #     layer_idx=6,
    #     d_model=768,
    #     n_features=16384,
    #     seq_len=128,
    #     token_batch_size=8,
    #     activation_batch_size=1024,
    #     max_examples_tinystories=5000,
    #     max_tokens_shakespeare=2_000_000,
    #     max_activation_batches=100,
    #     top_k=32,
    #     rank=4,
    #     device=device,
    # )