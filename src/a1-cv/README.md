# CSED 504 · A1 — CNNs vs. Vision Transformers

**Do convolutional networks (CNNs) or Vision Transformers (ViTs) make better image classifiers — and
does the answer change as you add more training data?**

This project trains both families of models on three datasets of the same tiny 32×32 images, stepping
from small to large — CIFAR-10 (50k images) → CIFAR-100 (50k images, harder) → ImageNet-32 (1.28M
images) — and measures not only which is more *accurate*, but what each one *costs* to train.

### The result: a crossover

| dataset | images / classes | best CNN | best ViT | winner |
|---|---|---|---|---|
| CIFAR-10 | 50k / 10 | **92.7%** | 85.1% | CNN |
| CIFAR-100 | 50k / 100 | **74.3%** | 62.4% | CNN |
| ImageNet-32 | 1.28M / 1000 | 41.7% | **43.0%** | ViT |

*(top-1 accuracy)*

A CNN is built with the assumption that nearby pixels belong together, which is a big head start when
data is scarce. A ViT has no such assumption — it has to *learn* that structure from the data — so it
only pulls ahead once there's a lot of data. That flip, from CNN winning to ViT winning, is the
**crossover**. Accuracy is only half the story, though: the ViTs cost several times more to train, and
scaling a ViT up can make it *worse*. The full picture is in **`report_crossover.ipynb`**.

---

## Setup (once)

Everything here runs in the shared **`uw-csed504`** conda environment. If you don't have it yet, the
top-level [`../../README.md`](../../README.md) has one-command setup scripts for Windows, macOS, and
Linux. Once it's installed:

```bash
conda activate uw-csed504
```

and select the **"Python (uw-csed504)"** kernel when you open a notebook.

---

## The fastest way in: use a trained model

You do **not** need a GPU, the datasets, or to run any training to use the results. Every trained model
is published on the [**`models-v1` release**](https://github.com/TrueRottweiler/WashingtonCsed504/releases/tag/models-v1),
and one function downloads whichever you ask for:

```python
# run this from inside src/a1-cv/, with the uw-csed504 environment active
from models import load_model

net = load_model("cifar100/vit")     # downloads the weights once, caches them, returns the model
```

`net` is an ordinary PyTorch model in eval mode — feed it a batch of 32×32 images and it classifies
them. The eight available tags:

- `imagenet32/resnet18`, `imagenet32/resnet50`, `imagenet32/vit`, `imagenet32/vit_base`
- `cifar10/resnet18`, `cifar10/vit`
- `cifar100/resnet18`, `cifar100/vit`

The weights live on the release rather than in git because they're large (45–350 MB each); `load_model`
keeps its downloads in a local `weights/` folder that git ignores.

---

## Explore in the notebooks

Each notebook runs top-to-bottom in **a minute or two** — they use a small, quick configuration so you
can watch the whole thing work. Open one, pick the `uw-csed504` kernel, and **Run All**.

| notebook | what it is |
|---|---|
| **`report_crossover.ipynb`** | **Start here.** The findings: the crossover table above, the accuracy-vs-training-cost tradeoff, and how to load the models. Reads results from disk, so it re-runs instantly. |
| `report_factory_performance.ipynb` | The engineering journey — how a plain coursework training loop was made fast enough to do this study, with the measured war-stories along the way. |
| `cifar10_train.ipynb`, `cifar100_train.ipynb`, `imagenet32_train.ipynb` | One per dataset: watch a CNN and a ViT actually train, and (for ImageNet) read the scoreboard. |

> These notebooks are **fast sanity checks** — a few epochs on a subset, just to see it work end to
> end. The full, hours-long training that produced the published numbers runs from the command line,
> covered in [Training it yourself](#training-it-yourself) below. Both the notebooks and the command
> line import the same `models.py` / `cifar_data.py` / `train_loop.py`, so there is one copy of the
> training code and nothing to keep in sync.

---

## What's in this folder

**The notebooks** are listed above. Behind them:

| file | what it does |
|---|---|
| `models.py` | The two architectures — ResNet-18/50 and ViT / ViT-base — each given a stem that works on 32×32 images. Also home to `load_model`. |
| `cifar_data.py`, `imagenet_data.py` | Load a dataset **onto the GPU once** and hand back augmented batches from there — no per-batch copying (see [How it's fast](#how-its-fast-the-one-idea-worth-knowing)). |
| `imagenet_prepare.py` | A one-time step that unpacks ImageNet-32 into fast flat arrays. Only needed if you want to *train* on ImageNet. |
| `train_loop.py` | The training loop shared by every run: forward/backward, accuracy metrics, checkpointing, logging. |
| `train_run.py` | Trains **one** model on **one** GPU. |
| `train_fleet.py` | Trains **several** models across **both** GPUs at once. |
| `dashboard.py` | A live, read-only terminal view of every training run in progress — accuracy curves, throughput, ETA. |
| `perf/` | A small tool that estimates how long a training run will take *before* you launch it. |

**Created while you run things (all git-ignored, all regenerable):**

```
data/      the datasets, downloaded on first use
runs/      each run's checkpoint (.pt) and per-epoch metrics (.jsonl) — the reports read these
logs/      each run's console output
weights/   trained models that load_model has downloaded
```

---

## Training it yourself

You only need this section to **reproduce** the training. To just *use* a model, use
[`load_model`](#the-fastest-way-in-use-a-trained-model) above — no training required.

All real training runs from the terminal (never from a notebook, so quick experiments and long runs
never get confused with each other). There are just **two commands**, and both save their results to
`runs/` where the reports can read them.

### 1. Train one model — `train_run.py`

You pick a **dataset** and a **model**; the training recipe (optimizer, learning rate, data
augmentation, gradient clipping) is chosen automatically to suit the model, so you don't have to know
those details.

```bash
# First, a quick wiring check — runs a couple of tiny epochs, then exits:
python train_run.py --dataset cifar100 --model resnet18 --smoke-test

# Then the real run:
python train_run.py --dataset cifar100 --model resnet18 --epochs 40
python train_run.py --dataset cifar10  --model vit      --epochs 200 --gpu 1
```

The flags:

- **`--dataset`** — `cifar10`, `cifar100`, or `imagenet32`. CIFAR downloads itself the first time you
  use it; ImageNet needs the one-time prep at the end of this section.
- **`--model`** — `resnet18`, `resnet50`, `vit`, or `vit_base`.
- **`--epochs`** — how long to train. CNNs converge quickly (30–40 epochs); ViTs need many more
  (around 200) to reach their best.
- **`--gpu N`** — which GPU to use (default `0`). **`--resume`** picks up from the last checkpoint if a
  run was interrupted.

### 2. Train a whole batch — `train_fleet.py`

The eight models in the results table weren't trained one at a time by hand. `train_fleet.py` trains a
batch of them, and on a two-GPU machine keeps both cards busy — one model per card, starting the next
as each finishes. It ships with the two batches that make up this study:

```bash
python train_fleet.py --queue cifar       # the four CIFAR models        (~30 min on two GPUs)
python train_fleet.py --queue imagenet    # the four ImageNet-32 models   (several hours)
python train_fleet.py --queue seeds       # both headline models, 80 epochs x 3 seeds (~5 h)
python train_fleet.py --queue overnight   # the above plus both capacity models  (~12 h)
```

`cifar` trains `resnet18` and `vit` on both CIFAR-10 and CIFAR-100; `imagenet` trains `resnet18`,
`resnet50`, `vit`, and `vit_base` on ImageNet-32. Between them they reproduce every number in the
results table. Each batch already carries the right schedule per model — the ViTs get 200 epochs (a
Transformer needs a long run to converge), the ResNets far fewer — so you don't have to remember them.
Add `--smoke` to prove the wiring in about a minute first.

`seeds` is different, and it settles two questions at once. It re-runs both headline models **at 80
epochs, under three seeds each**.

The first question is spread: the gap between the two arms is about a point, and two runs of one
configuration already differ by a few tenths, so a single run per arm can't tell you whether the gap is
real. The second is the epoch budget. Every CNN run peaks on its final epoch, which looks like it was
stopped early — but the learning rate anneals to zero over whatever budget it's given, so a late peak
is what that schedule always produces. Going from 40 to 60 epochs gained the CNN only 0.2 points;
the catch is that control used gradient clipping, while the CNN we quote runs without it, so the
headline configuration has never had an epoch control. Doubling the budget answers it.

The repeats are tagged `resnet18_s1`, `vit_s1`, and so on, so they sit beside the originals rather than
overwriting them, and the ImageNet-32 notebook summarizes them as a mean and a spread — grouped by
budget, so the 80-epoch runs are never averaged in with the 40-epoch ones.

On a single-GPU laptop you don't need the fleet at all — run the models one at a time with
`train_run.py` (step 1).

### 3. Watch it live — `dashboard.py`

From a **second** terminal, while training is running:

```bash
python dashboard.py
```

It shows one panel per run — its progress, accuracy curve, throughput, and both a predicted and a live
ETA — plus GPU utilization. It only reads files, so it's safe to open and close any time.

### One-time ImageNet-32 setup

CIFAR downloads itself automatically. ImageNet-32 is different: it ships as JPEG images packed inside
parquet files, which is far too slow to decode on every batch. So we decode it once, up front, into
flat arrays (~3.9 GB on disk):

```bash
git clone https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32   # accept the ImageNet terms first
python imagenet_prepare.py
```

You only need this if you're training on ImageNet — not to use a published `imagenet32/*` model.

---

## Adding your own experiment

The baseline is meant to be built on. Everything here is additive — you shouldn't need to move or
rename an existing file.

**1. Add a model.** Architectures live in [`models.py`](models.py). Write a builder that takes
`num_classes` and returns an `nn.Module`, then register it in the `BUILDERS` dict:

```python
BUILDERS = {
    ...
    'vit_small': make_vit_small,
}
```

That name is immediately a valid `--model` everywhere: `train_run.py`, `train_fleet.py`, and the
notebooks all read the same dict, so there is nothing else to wire up.

**Name it deliberately** — the recipe is chosen from the model's *name*, not from a config you have to
remember:

- a name starting with `vit` gets AdamW and the strong augmentation a Transformer needs
- anything else gets SGD with momentum and no mixup
- gradient clipping defaults on, except for exactly `resnet18`, where it cost about five points

So `vit_small` inherits the Transformer recipe for free, while `myresnet` inherits the CNN one. Any of
it can be overridden with `--strong-aug` / `--no-strong-aug`, `--clip`, `--lr`, and the rest.

**2. Train it.** Prove the wiring first, then run it for real:

```bash
python train_run.py --dataset cifar100 --model vit_small --smoke-test   # ~1 min, then exits
python train_run.py --dataset cifar100 --model vit_small --epochs 40
```

**3. Repeat it before you believe it.** One run is one sample, and two runs of the same configuration
here land a few tenths of a point apart. `--seed` re-runs it under a different seed; give the repeat
its own `--tag` so it sits beside the original instead of overwriting it:

```bash
python train_run.py --dataset cifar100 --model vit_small --epochs 40 --seed 2 --tag vit_small_s2
```

**4. Where the results go.** Each run appends `runs/<tag>.jsonl` (one line per epoch) and writes
`runs/<tag>_result.json` when it finishes. The report notebooks read those files off disk, so a new run
shows up in the tables without editing a notebook. Commit the small `.jsonl` and `_result.json`; the
checkpoint (`runs/<tag>.pt`), the datasets, and the logs are git-ignored on purpose, so it's hard to
accidentally push a few hundred MB.

**Keep the notebooks fast.** They are pipeline checks that finish in a minute or two. Long training
belongs on the command line, where an hours-long run can't be confused with an interactive experiment —
then let the report notebooks read the result off disk.

---

## How it's fast (the one idea worth knowing)

At 32×32, a modern GPU can train these small models faster than an ordinary data pipeline can feed it —
the usual PyTorch `DataLoader` with CPU workers becomes the bottleneck, and the GPU sits idle waiting
for the next batch. So we flip it around: **load the entire dataset into GPU memory once, and do the
image augmentation on the GPU too.** No workers, no per-batch copying from CPU to GPU, nothing for the
CPU to do in the training loop.

This only works because the images are tiny — ImageNet-32's whole training set is 3.9 GB as `uint8`,
which fits comfortably in a modern GPU's memory. You could never do this with the full 224×224 ImageNet
(hundreds of GB). `report_factory_performance.ipynb` walks through this trick and the other speedups
that made the study practical.
