# CSED 504 — A1 Computer Vision

**CNNs vs. Transformers, and what data scale does to the answer** — one training pipeline, made fast
on every machine (Mac / Windows / Linux / Colab), taken from CIFAR-10 (50k images) up to ImageNet-32
(1.28M images), for both CNNs and Vision Transformers.

The scientific write-up — results, controls, the crossover, and the engineering notes — lives one
level up in [`../README.md`](../README.md). **This file is the map of the folder: what each file is,
and where to start.**

## Start here

1. **[`report_factory_performance.ipynb`](report_factory_performance.ipynb)** — the capstone. The whole
   journey in one notebook: how we took the coursework training loop and made it fast, the measured
   war-stories (feeding the GPU, the flip that stalled the pipeline, mixed precision × memory format),
   the CNN-vs-Transformer crossover, and predicting a run's cost before launching it. Runs top to bottom
   in a couple of minutes on `MODE='fast'`. **Read this one first.**
2. Then the three per-dataset training notebooks below, in increasing scale.

## Notebooks

| notebook | what it does |
|---|---|
| `report_factory_performance.ipynb` | the capstone / journey — **start here** |
| `cifar10_train.ipynb` | CIFAR-10 (50k, 10 classes) — the small-data study, self-contained |
| `cifar100_train.ipynb` | CIFAR-100 from the HuggingFace Hub — a second data source through the same models |
| `imagenet32_train.ipynb` | ImageNet-32 (1.28M, 1000 classes) — the scoreboard + crossover, and how to launch the real run |

Each dataset notebook is a **fast** interactive check (a small subset, a few epochs — a minute or two)
so you can watch it work end to end. The full, expensive training runs from the terminal (below), never
in a notebook, so the two never get conflated; the reports read whichever results are on disk.

## Get the trained models — no training required

Every trained model is published on the **[`models-v1` release](https://github.com/TrueRottweiler/WashingtonCsed504/releases/tag/models-v1)** (weights only, ~45–350 MB each). You do **not** clone them (they're too big for git) and you do **not** retrain — `load_model` downloads the one you ask for, caches it locally, and hands back a ready-to-eval model. No GPU needed.

```python
from models import load_model          # run from inside src/a1-cv/ (with the uw-csed504 env active)
net = load_model("cifar100/vit")       # downloads once, returns a ready nn.Module in eval mode
```

Tags: `imagenet32/{resnet18,resnet50,vit,vit_base}`, `cifar10/{resnet18,vit}`, `cifar100/{resnet18,vit}`. Top-1 accuracy:

| | CIFAR-10 | CIFAR-100 | ImageNet-32 |
|---|---|---|---|
| **CNN** (ResNet) | 92.7% | 74.3% | 41.7% |
| **ViT** | 85.1% | 62.4% | 43.0% |

The **crossover**: the CNN wins on small/medium data, the ViT only once the data is large (1000 classes, 1.28M images). See `report_crossover.ipynb` for the cost/benefit story behind these numbers.

## Python — the shared machinery

| file | role |
|---|---|
| `models.py` | the architectures: ResNet-18/50 and ViT / ViT-base, each given a 32×32-friendly stem |
| `cifar_data.py` | GPU-resident loader for small in-memory images — CIFAR-10 **and** CIFAR-100 |
| `imagenet_data.py` | GPU-resident loader for ImageNet-32 (memory-maps the prepared arrays; GPU-side augmentation) |
| `imagenet_prepare.py` | **one-time** decode of HuggingFace ImageNet-1k-32×32 (JPEG-in-parquet) into flat `.npy` arrays |
| `train_loop.py` | the epoch loop, top-1/top-5 metrics, checkpointing, JSONL logging — shared by every run |
| `train_run.py` | CLI: **one** training run on one GPU |
| `train_fleet.py` | fills **both** GPUs — one model per card, concurrently, until a queue drains |
| `dashboard.py` | live terminal dashboard: both cards, every run's curves + ETA (read-only) |
| `perf/` | the training-cost estimator (`perfkit.py`), a headless collector (`collect.py`), its notebook, and the results DB |

Why two data loaders? `cifar_data.py` takes images already in memory (torchvision hands you decoded
arrays); `imagenet_data.py` memory-maps 3.9 GB of prepared `uint8` off disk. Only ImageNet needs the
separate `imagenet_prepare.py` step, because it ships as JPEG-encoded blobs inside parquet and is far too
big to decode per batch — CIFAR arrives ready to use.

## Data & outputs (git-ignored, all regenerable)

```
data/
  cifar10/     torchvision CIFAR-10   (auto-downloads)
  cifar100/    torchvision CIFAR-100  (auto-downloads)
  imagenet32/  prepared .npy arrays   (3.9 GB — see imagenet_prepare.py)
runs/          checkpoints + per-epoch metrics (one set per run name)
logs/          per-run stdout
```

## Running the real training (from the console)

All full training — every dataset — runs from the terminal through one trainer, so it never conflates
with the fast notebooks. `--dataset` picks the data; the recipe (optimizer, augmentation, LR, gradient
clipping) follows the model automatically.

```bash
# CIFAR (data auto-downloads from HuggingFace on first use):
python src/a1-cv/train_run.py --dataset cifar100 --model vit --epochs 200
#   ...or the whole CIFAR retraining across both GPUs in one command (both ViTs get 200 epochs):
python src/a1-cv/train_fleet.py --queue retrain

# ImageNet-32 needs its 3.9 GB prepared once first:
git clone https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32   # accept the ImageNet terms
python src/a1-cv/imagenet_prepare.py
python src/a1-cv/train_fleet.py                       # the ImageNet-32 capstone on both cards

# watch any run live, from another terminal:
python src/a1-cv/dashboard.py
```

`--smoke-test` runs a quick sanity check and exits — run it first. `--resume` picks up from the last
checkpoint. Results land in `runs/`; the report notebooks read them.

> **Teammates don't need any of this.** To *use* a trained model, skip straight to `load_model` above —
> no dataset, no GPU, no training.

## Shared / elsewhere
- Device detection (`get_device`, seeds, multi-GPU selection) is `../common/gpu_check.py` — shared with
  the other assignments, so it stays out of this folder.
- Superseded experiments and one-off scripts were moved to `../archive/`.
