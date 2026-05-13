import subprocess
import sys
from pathlib import Path

TASKS = ["identity", "square", "sin", "gaussian"]
BASE_DIR = Path(__file__).resolve().parent


def run(args):
    subprocess.run([sys.executable, *args], cwd=BASE_DIR, check=True)


for task in TASKS:
    print(f"=== Training {task} ===")
    run(["train_transformer.py", "--task", task, "--steps", "10000"])
    print(f"=== Analyzing geometry {task} ===")
    run(["analyze_geometry.py", "--task", task])
    print(f"=== Writing interactive PCA {task} ===")
    run(["interactive_pca.py", "--task", task, "--method", "pca"])
    print(f"=== Writing interactive UMAP {task} ===")
    run(["interactive_pca.py", "--task", task, "--method", "umap"])
    print(f"=== Comparing clusters and accuracy {task} ===")
    run(["compare_cluster_accuracy.py", "--task", task])
    ckpt = f"checkpoints/{task}/step_10000.pt"
    print(f"=== Training SAE {task} ===")
    run(["train_sae.py", "--task", task, "--checkpoint", ckpt])
    sae = f"checkpoints/{task}_sae_step_10000.pt"
    print(f"=== Analyzing SAE {task} ===")
    run(["analyze_sae.py", "--task", task, "--checkpoint", ckpt, "--sae", sae])
