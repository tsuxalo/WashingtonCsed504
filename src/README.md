# CSED 504 — A1 Computer Vision: CNNs vs. Transformers, and What Data Scale Does to the Answer

**Status: main training complete. The headline result SURVIVED its controls (see
[§3](#3-did-we-rig-the-race-the-controls)). Two baseline re-runs are still in flight to settle one
methodology wart, noted honestly below.**

---

## 1. The question

There are two families of image models, and they make opposite bets.

A **CNN** (ResNet) has an assumption hardwired into its structure: *nearby pixels belong together.* It
slides a small filter across the image, so it automatically knows that a cat is a cat whether it sits
in the corner or the middle. Nobody teaches it this — it is in the wiring. In ML jargon these free
assumptions are called **inductive biases**.

A **Vision Transformer** (ViT) has almost none of that. It cuts the image into tiles ("patches"),
treats each as a token, and lets every token attend to every other token. It does not even know that
two tiles are adjacent — that has to be *learned* from data.

That is the trade:

| | CNN | Transformer |
|---|---|---|
| Assumptions built in | strong (locality, translation invariance) | almost none |
| Must learn spatial structure from data | no | **yes** |
| On a small dataset | strong | weak — not enough data |
| On a huge dataset | plateaus | **overtakes** — it learns better assumptions than we hardcoded |

**So: where is the crossover?** That is the whole project.

- `a1-cv/cifar10_train.ipynb` — **CIFAR-10, 50,000 images, 10 classes.** The small-data side.
- `a1-imagenet32/` — **ImageNet-1k downsampled to 32×32: 1,281,167 images, 1000 classes.** 25× more data,
  *identical image size*, so the models port over with no architectural change at all.

---

## 2. Results so far

### CIFAR-10 (50k images) — finished

| model | params | epochs | test top-1 |
|---|---|---|---|
| ResNet-18 (CNN) | 11.2M | 20 | **92.50%** |
| ViT (CNN's recipe) | 10.7M | 20 | 55.18% |
| ViT (proper ViT recipe) | 10.7M | 60 | 73.03% |

**The CNN wins by ~19 points.** Note the middle row: the ViT trained with the CNN's own recipe scores
55%, and *the same model* with a transformer recipe scores 73%. Nothing about the architecture changed
between those two rows — only the optimizer, schedule, and augmentation. **Most of the ViT's apparent
weakness is recipe, not architecture.** The remaining gap is the real cost of having no inductive bias
on a small dataset.

### ImageNet-32 (1.28M images) — complete

| run | family | params | epochs | mixup | val top-1 | val top-5 |
|---|---|---|---|---|---|---|
| **vit** | ViT | 11.1M | 60 | yes | **42.76%** | 66.33% |
| **vit_40ep** | ViT | 11.1M | 40 | yes | **42.33%** | 66.68% |
| vit_base | ViT | 85.9M | 40 | yes | 41.56% | 63.49% |
| **resnet18** | CNN | 11.7M | 40 | no | **41.54%** | 66.86% |
| resnet50 | CNN | 25.5M | 40 | no | 41.32% | 66.19% |
| resnet18_60ep | CNN | 11.7M | 60 | no | 37.06% | 62.58% |
| resnet18_aug60 | CNN | 11.7M | 60 | **yes** | 33.91% | 59.43% |

**Every ViT configuration beat every CNN configuration.** Best ViT 42.76%, best CNN 41.54%.

Reference point: the paper that created this dataset ([Chrabaszcz et al.
2017](https://arxiv.org/abs/1707.08819)) reports **59.0% top-1 / 81.1% top-5** for a much larger
WRN-28-10 trained longer. So 41–43% from an 11M-parameter model in 40–60 epochs is a sane number.
Random guessing on 1000 classes is 0.1%.

### The crossover

```
CIFAR-10     (50k):    CNN 92.5%   |  ViT 73.0%   ->  CNN ahead by 19.5 points
ImageNet-32  (1.28M):  CNN 41.5%   |  ViT 42.8%   ->  ViT ahead by  1.2 points
```

(The absolute numbers are not comparable across datasets — ImageNet-32 has 1000 classes, so everything
is lower. What *is* comparable is the **sign of the gap**: the CNN leads on the left, the ViT leads on
the right.)

**This is the crossover the ViT literature predicts, reproduced on our own hardware in a
parameter-matched comparison.** 25× more data flipped the winner.

---

## 3. Did we rig the race? The controls

The winning ViT was given **two advantages the CNN never got**: 60 epochs instead of 40, and heavy
augmentation (mixup + CutMix + random erasing) instead of just crop+flip. A 1.2-point win, with two
uncontrolled advantages, is not a result. So we ran the controls.

**Why the ViT got those advantages at all:** our first ImageNet-32 ViT run used the CNN's light
augmentation, and the ViT **memorized the training set** — 97.4% train accuracy against 32.6%
validation, a 65-point gap, on 1.28 *million* images. The ResNet, with the same light augmentation,
stayed healthy (+8.7% gap).

### What the controls found

| control | result | what it says |
|---|---|---|
| **`vit_40ep`** — ViT cut to the CNN's 40 epochs | **42.33%** — still beats the CNN's 41.54% | The win is **not** bought with extra epochs. |
| **`resnet18_60ep`** — CNN given the ViT's 60 epochs | 37.06% | More epochs did not help the CNN. |
| **`resnet18_aug60`** — CNN given 60 epochs **and** mixup | **33.91%** | Giving the CNN the ViT's exact advantages made it **7.6 points WORSE.** |

**The headline survives, and the decisive control broke the opposite way from what we expected.**

### The deepest finding: augmentation is a *substitute* for inductive bias

Mixup **helps the Transformer enormously and actively hurts the CNN** (33.91% vs 37.06%, identical
except for mixup). That is not a quirk — it follows directly from the architecture:

- The ViT has **no inductive bias to restrain it.** Given only crop+flip, it memorized 1.28M images.
  Heavy augmentation is what stops it — it *replaces* the constraint the architecture doesn't provide.
- The CNN **cannot easily memorize.** Weight sharing and locality physically limit what it can
  represent, so it is *already* regularized by its own structure. Pile mixup on top and you are simply
  making its job harder for no benefit.

So heavy augmentation is not a free advantage we forgot to share. It is a tool for a problem **only the
Transformer has.**

### Scale was not the lever — data was

- `vit_base` (85.9M params, 8× the small ViT) scored **41.56%** — it did **not** beat the 11.1M ViT.
- `resnet50` (25.5M params, 2× ResNet-18) scored **41.32%** — it did **not** beat ResNet-18.

Both bigger models were still improving when their schedules ended, so this is a statement about our
**compute budget**, not proof that scaling stops working. But within this budget: **more parameters
bought nothing. More *data* flipped the result.**

---

## 4. The models

All four see identical data and are adapted for 32×32 inputs the same way. The stock torchvision models
target 224×224 ImageNet and would destroy a 32×32 image, so:

- **ResNets:** replace the 7×7 stride-2 stem conv with a 3×3 stride-1 conv, and delete the early
  max-pool. (Otherwise a 32×32 image is crushed to 8×8 before any real work happens.)
- **ViTs:** `patch_size=4`, giving an 8×8 grid = **64 tokens**. The ImageNet default of 16 would give a
  2×2 grid — *four* tokens for a whole image, which is useless.

Note these are the *same* adaptations used in the CIFAR notebook. Because ImageNet-32 is also 32×32,
**the models port over with no changes at all** — only `num_classes` moves from 10 to 1000.

| | resnet18 | resnet50 | vit | vit_base |
|---|---|---|---|---|
| family | CNN | CNN | Transformer | Transformer |
| params | 11.7M | 25.5M | 11.1M | 85.9M |
| depth | 18 layers | 50 layers | 6 blocks | 12 blocks |
| width | — | — | 384, 6 heads | 768, 12 heads |
| throughput | 17.7k img/s | 6.3k img/s | 13.7k img/s | 2.8k img/s |
| **question it answers** | the baseline | is it just "bigger CNN wins"? | **the crossover** | does scaling the ViT help? |

`resnet18` (11.7M) and `vit` (11.1M) are deliberately **parameter-matched** — that is the fair fight;
neither can win by being bigger.

---

## 5. The training recipes

The two families are trained differently, on purpose. This is not favoritism; it is what each needs.

| | CNNs | Transformers |
|---|---|---|
| optimizer | SGD + Nesterov momentum 0.9 | **AdamW** |
| learning rate | **0.2** (= 0.1 × batch/256) | **0.001** |
| weight decay | 5e-4 | 0.05 |
| augmentation | random crop + horizontal flip | crop + flip **+ mixup + CutMix + random erasing** |
| gradient clipping | 1.0 | 1.0 |

Identical for everything: batch 512, 5-epoch linear **warmup** → cosine decay, label smoothing 0.1,
mixed precision (fp16 autocast + GradScaler), and the same 1.28M images.

Two notes for anyone reading the code:

- **Warmup is not optional for a transformer.** Without it the attention softmax saturates on the first
  few noisy batches and the model lands in a bad place it never escapes. (This is why minGPT's trainer
  in CSED 503 had a warmup schedule.)
- **AdamW's `weight_decay=0.05` is not comparable to SGD's `5e-4`.** AdamW *decouples* decay from the
  gradient, so the number means something different. Do not read across.

---

## 6. Repo layout & how to run it

```
src/
  a1-cv/
    cifar10_train.ipynb      the CIFAR-10 study (self-contained; runs anywhere with PyTorch)
  a1-imagenet32/
    prepare_data.py          ONE-TIME: parquet/JPEG -> raw uint8 arrays (~30 seconds)
    data.py                  GPU-resident dataset + GPU-side augmentation
    models.py                the four architectures
    engine.py                train/eval loop, top-1/top-5, checkpointing, JSONL logging
    train.py                 CLI: one training run
    scheduler.py             keeps both GPUs busy with a queue of runs
    monitor.py               live dashboard (read-only)
    data/                    generated arrays (4.1 GB, gitignored)
    runs/                    checkpoints + per-epoch metrics (JSONL)
    logs/                    stdout of each run
```

### Setup

```bash
# 1. get the dataset (gated -- you must accept the ImageNet terms on HuggingFace first)
git clone https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32

# 2. decode it once into raw arrays (~30s, writes 4.1 GB)
python src/a1-imagenet32/prepare_data.py

# 3. train
python src/a1-imagenet32/train.py --model resnet18 --gpu 0 --epochs 40
python src/a1-imagenet32/train.py --model vit      --gpu 1 --epochs 60

# 4. watch it
python src/a1-imagenet32/monitor.py
```

`--smoke-test` runs a 30-second sanity check on a small subset and exits. **Always run it first.**
`--resume` picks up from the last checkpoint (written every epoch).

---

## 7. Engineering notes — things we measured that surprised us

These cost us real time. They are written down so nobody repeats them.

**The whole dataset lives in GPU memory.** 1.28M images at 32×32 uint8 is only **3.9 GB**, and the card
has 96 GB. So we upload it once and generate batches *on the GPU* — augmentation is a dozen lines of
tensor math. No DataLoader, no worker processes, no PIL, no per-batch host→device copies. This is only
possible because the images are tiny (224×224 ImageNet would be ~190 GB).

**`num_workers=0` on Windows is a myth that costs 3×.** The old "workers hang in Jupyter on Windows"
rule is obsolete on PyTorch 2.x. On CIFAR, going 0 → 8 workers took an epoch from 14.2s to 4.7s. Also:
`persistent_workers=True` is *mandatory*, or the DataLoader respawns every worker at each epoch boundary
and eats the entire gain.

**A GPU can be starved by a single CPU core.** Python/PIL augmentation runs ~4,000 img/s per core; the
GPU trains at ~13,000 img/s. Before optimizing anything, find out **which side is waiting**. Low
`nvidia-smi` utilization during training means you are CPU-bound.

**`channels_last` is 3× SLOWER at 32×32.** The standard "free speedup for convnets" advice. It needs
ImageNet-sized spatial dimensions to pay off; at CIFAR scale it is all overhead. Measure, don't copy.

**Learning-rate scaling: `lr = 0.1 × batch/256`, not `/128`.** We used the CIFAR baseline (0.1 @ 128)
and got lr=0.4 at batch 512 — double the correct value. ResNet-18 slowly diverged (accuracy *peaked at
epoch 2* then fell) and **ResNet-50 went to `loss = NaN` on epoch 1**. A 2× LR error is not a tuning
detail; it is the difference between training and not training. `engine.py` now aborts immediately on a
NaN loss.

**Deep ResNets at large batch need `zero_init_residual=True`.** ResNet-50 hit NaN at *both* lr 0.4 and
lr 0.2. Initializing each residual block as an identity map (the standard Goyal et al. large-batch
trick) fixed it.

**On a GPU, a Python loop over samples is the enemy.** Our first `random_erasing_` looped over the ~25%
of the batch being erased and dropped ViT throughput from 14.3k to 4.2k img/s. Rewriting it as a
broadcast mask restored 15.3k. Same root cause as the point above about kernel launches.

**Two jobs per GPU does NOT double throughput.** We tried it. The cards are power-limited (300/300 W);
a second process just splits the same throughput and doubles everyone's wall-clock. **One job per GPU,
two GPUs = two experiments in parallel** — that is what the second card is actually for. `DataParallel`
across both cards measured **0.98×** at this model scale and is not worth it.

**Augmented train accuracy is not comparable to clean test accuracy.** They are two different exams. With
heavy augmentation the *training* number can even come out **below** the test number, which looks
impossible. Do not read the train/val gap off the training printout — score the training set with the
*test* transform if you want the real overfitting number. This trap bit us three separate times.

---

## 8. Known issues / open questions

- **We changed the CNN mid-experiment — this is a real methodology wart.** `zero_init_residual=True`
  and gradient clipping were added to `make_resnet18` while fixing ResNet-50's NaN divergence, *after*
  the original `resnet18` baseline had already run. So `resnet18` (41.54%) and its own controls
  (`resnet18_60ep`, `resnet18_aug60`) were **not built from identical code**, which is exactly the sin
  those controls existed to prevent. That may explain why `resnet18_60ep` (more training) came out
  *worse* than the baseline — more training should not hurt. Two re-runs (`resnet18_v2` with the
  current code, `resnet18_noclip` with clipping disabled) are measuring what that change cost. **The
  ViT-vs-CNN conclusion does not depend on it** — the ViT beats the *best* CNN number under any
  version of the code — but the CNN-vs-CNN rows should be treated as provisional.
- **`resnet50` recovered.** It hit `loss = NaN` on epoch 1 at both lr 0.4 and lr 0.2;
  `zero_init_residual=True` fixed it and it finished at 41.32%, on par with ResNet-18.
- **Nothing here is converged.** Every run was still improving when its schedule ended. Longer
  schedules would raise all of these numbers, and would likely help `vit_base` the most (86M params in
  40 epochs is badly undertrained).
- **One seed per config.** We have not measured run-to-run variance, so a ~1-point difference is
  suggestive, not decisive. The ViT's 1.2-point margin deserves a repeat with a second seed.
- We report accuracy on the official **50k validation split**. ImageNet's 100k "test" split has no
  usable public labels.

## 9. What to look at first

- The CIFAR notebook, `a1-cv/cifar10_train.ipynb` — it explains the whole pipeline, cites the CSED 502
  NumPy code that each PyTorch call replaces, and contains the CNN-vs-ViT comparison at small scale.
- `a1-imagenet32/data.py` — the GPU-resident dataset. This is the most unusual thing we built.
- `a1-imagenet32/results.ipynb` — **the results.** Reads the per-epoch metrics off disk, prints the
  scoreboard, plots the learning curves and the crossover. It does no training; re-run it any time.
- `a1-imagenet32/monitor.py` — the live dashboard. Run it while training. Shows both GPUs and every run.
