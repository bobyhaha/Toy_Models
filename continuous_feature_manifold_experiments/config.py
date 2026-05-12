from dataclasses import dataclass

@dataclass
class ModelConfig:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    d_mlp: int = 512
    dropout: float = 0.0
    max_seq_len: int = 64

@dataclass
class TrainConfig:
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-2
    steps: int = 20000
    eval_every: int = 500
    save_every: int = 1000
    precision: int = 3
    x_min: float = -3.14159
    x_max: float = 3.14159
    train_size: int = 100000
    test_size: int = 10000
    seed: int = 0
