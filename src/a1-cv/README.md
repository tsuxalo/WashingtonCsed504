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

Each notebook is meant to run **`fast`** in-notebook (a small subset, a few epochs — minutes) so you can
watch it work end to end. The full multi-hour **`perf`** training is driven from the terminal (below),
and the report reads whichever results are on disk.

> The shared `fast` / `perf` config knob is already in `report_factory_performance.ipynb`; wiring the
> same knob into the three dataset notebooks (so one `train_run.py` does every dataset's `perf` run
> overnight) is the next step.

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

## Running the real (`perf`) training

CIFAR is small enough to just run its notebook. ImageNet-32 takes hours, so it runs from the terminal:

```bash
# 1. one-time: get + decode the dataset (writes data/imagenet32/, ~3.9 GB)
git clone https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32   # accept the ImageNet terms first
python src/a1-cv/imagenet_prepare.py

# 2. train — one card each, concurrently...
python src/a1-cv/train_run.py --model resnet18 --gpu 0 --epochs 40
python src/a1-cv/train_run.py --model vit      --gpu 1 --epochs 60
#    ...or fill both cards from a queue:
python src/a1-cv/train_fleet.py

# 3. watch it (from another terminal)
python src/a1-cv/dashboard.py
```

`--smoke-test` runs a 30-second sanity check and exits — always run it first. `--resume` picks up from
the last checkpoint. Results land in `runs/`; open `imagenet32_train.ipynb` to read the scoreboard and
plot the crossover.

## Shared / elsewhere
- Device detection (`get_device`, seeds, multi-GPU selection) is `../common/gpu_check.py` — shared with
  the other assignments, so it stays out of this folder.
- Superseded experiments and one-off scripts were moved to `../archive/`.
