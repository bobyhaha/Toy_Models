import subprocess
import sys

TASKS = ["identity", "square", "sin", "gaussian"]

for task in TASKS:
    print(f"=== Training {task} ===")
    subprocess.run([sys.executable, "train_transformer.py", "--task", task, "--steps", "10000"], check=True)
    print(f"=== Analyzing geometry {task} ===")
    subprocess.run([sys.executable, "analyze_geometry.py", "--task", task], check=True)
    ckpt = f"checkpoints/{task}/step_10000.pt"
    print(f"=== Training SAE {task} ===")
    subprocess.run([sys.executable, "train_sae.py", "--task", task, "--checkpoint", ckpt], check=True)
    sae = f"checkpoints/{task}_sae_step_10000.pt"
    print(f"=== Analyzing SAE {task} ===")
    subprocess.run([sys.executable, "analyze_sae.py", "--task", task, "--checkpoint", ckpt, "--sae", sae], check=True)
