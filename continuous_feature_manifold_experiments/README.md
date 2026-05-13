# Continuous Feature Manifold Experiments

This repo tests whether small transformers trained on continuous decimal-string tasks develop smooth low-dimensional representations of continuous variables.

## Experiments

Tasks:
- `identity`: y = x
- `square`: y = x^2
- `sin`: y = sin(x)
- `gaussian`: y = exp(-x^2)

Main analyses:
- Train a small decoder-only transformer on decimal-string regression.
- Save checkpoints every 10 steps through step 500, every 50 steps through step 2500, then every 100 steps.
- Collect residual stream activations at the position right before generating y.
- Measure whether h(x) becomes smooth and low-dimensional over training.
- Train sparse autoencoders on h(x).
- Analyze whether SAE features tile the continuous manifold.

## Setup

```bash
pip install torch numpy matplotlib plotly scikit-learn scipy tqdm umap-learn
```

## Run

Train all tasks:

```bash
python run_all.py
```

Or train one task:

```bash
python train_transformer.py --task sin --steps 20000 --device cuda
```

Analyze checkpoints:

```bash
python analyze_geometry.py --task sin --device cuda
```

Interactive PCA and UMAP over the latest training run:

```bash
python interactive_pca.py --task sin --method pca --device cuda
python interactive_pca.py --task sin --method umap --device cuda
```

Both interactive plots include a training-step slider, hoverable `x` values,
nearest-neighbor summaries, and the nearest logged test accuracy for that step.

Compare Experiment 1 cluster formation against model accuracy:

```bash
python compare_cluster_accuracy.py --task sin --device cuda
```

Train SAE:

```bash
python train_sae.py --task sin --checkpoint checkpoints/sin/step_20000.pt --device cuda
```

Analyze SAE:

```bash
python analyze_sae.py --task sin --checkpoint checkpoints/sin/step_20000.pt --sae checkpoints/sin_sae_step_20000.pt --device cuda
```

Plots are written to `plots/`.

## Research hypotheses

H1: During training, h(x) becomes smoother as a function of x.

H2: The intrinsic dimension of h(x) decreases toward a low-dimensional manifold.

H3: Accuracy improvements correlate with geometry transitions.

H4: SAEs represent continuous variables through multiple local sparse features, i.e. a tiling of the manifold, rather than one global scalar latent.

## Suggested first run

```bash
python train_transformer.py --task sin --steps 10000 --device cuda
python analyze_geometry.py --task sin --device cuda
python train_sae.py --task sin --checkpoint checkpoints/sin/step_10000.pt --device cuda
python analyze_sae.py --task sin --checkpoint checkpoints/sin/step_10000.pt --sae checkpoints/sin_sae_step_10000.pt --device cuda
```
