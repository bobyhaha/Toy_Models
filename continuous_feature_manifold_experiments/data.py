import math
import numpy as np
import torch
from torch.utils.data import Dataset
from tokenizer import CharTokenizer

def f_task(x, task):
    if task == "identity":
        return x
    if task == "square":
        return x * x
    if task == "sin":
        return math.sin(x)
    if task == "gaussian":
        return math.exp(-x * x)
    raise ValueError(f"Unknown task: {task}")

def format_num(x, precision=3):
    # fixed precision makes the grammar stable
    return f"{x:.{precision}f}"

def make_example(x, task, precision=3):
    y = f_task(float(x), task)
    prompt = f"x={format_num(x, precision)};y="
    target = format_num(y, precision)
    full = prompt + target
    return prompt, target, full, float(y)

class ContinuousFunctionDataset(Dataset):
    def __init__(
        self,
        task="sin",
        n=100000,
        x_min=-3.14159,
        x_max=3.14159,
        precision=3,
        seed=0,
    ):
        self.task = task
        self.n = n
        self.x_min = x_min
        self.x_max = x_max
        self.precision = precision
        rng = np.random.default_rng(seed)
        self.xs = rng.uniform(x_min, x_max, size=n).astype(np.float32)
        self.tokenizer = CharTokenizer()
        self.examples = [make_example(x, task, precision) for x in self.xs]
        self.max_len = max(len(self.tokenizer.encode(full)) for _, _, full, _ in self.examples)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        prompt, target, full, y = self.examples[idx]
        ids = self.tokenizer.encode(full)
        x = torch.full((self.max_len,), self.tokenizer.pad_id, dtype=torch.long)
        x[:len(ids)] = torch.tensor(ids, dtype=torch.long)

        # next-token targets
        inp = x[:-1].clone()
        labels = x[1:].clone()
        labels[labels == self.tokenizer.pad_id] = -100

        # Mask loss on prompt tokens; only train on y digits and eos.
        # Because input has BOS, next-token labels align one step ahead.
        prompt_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        labels[:len(prompt_ids)-1] = -100

        return {
            "input_ids": inp,
            "labels": labels,
            "x_value": torch.tensor(self.xs[idx], dtype=torch.float32),
            "y_value": torch.tensor(y, dtype=torch.float32),
        }

def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
        "x_value": torch.stack([b["x_value"] for b in batch]),
        "y_value": torch.stack([b["y_value"] for b in batch]),
    }

def make_grid_prompts(task, n=1000, x_min=-3.14159, x_max=3.14159, precision=3):
    tok = CharTokenizer()
    xs = np.linspace(x_min, x_max, n).astype(np.float32)
    prompts = []
    ys = []
    for x in xs:
        prompt, target, full, y = make_example(float(x), task, precision)
        prompts.append(prompt)
        ys.append(y)
    max_len = max(len(tok.encode(p, add_bos=True, add_eos=False)) for p in prompts)
    input_ids = torch.full((n, max_len), tok.pad_id, dtype=torch.long)
    last_positions = []
    for i, p in enumerate(prompts):
        ids = tok.encode(p, add_bos=True, add_eos=False)
        input_ids[i, :len(ids)] = torch.tensor(ids)
        last_positions.append(len(ids) - 1)
    return xs, np.array(ys, dtype=np.float32), input_ids, torch.tensor(last_positions, dtype=torch.long)
