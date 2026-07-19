# A1 · Computer Vision — CNNs vs. Vision Transformers

This is the write-up for Assignment 1 (Computer Vision); A2 covers NLP separately. For how to set up,
load a trained model, or run the training yourself, see [`a1-cv/README.md`](a1-cv/README.md). This file
is the *what we found, and why*.

## The question

Two families of image model make opposite bets.

A **CNN** (we use ResNet) has an assumption built into its wiring: nearby pixels belong together. It
slides a small filter across the image, so it recognizes a cat whether the cat sits in the corner or the
middle, without being taught to. Assumptions baked into an architecture like this are called *inductive
biases*, and they act like free training data.

A **Vision Transformer** (ViT) has almost none of that. It cuts the image into small patches, treats
each as a token, and lets every token attend to every other. It does not even start out knowing which
patches are neighbors — that has to be learned from the data.

So the CNN begins with a head start the ViT has to earn. The bet the ViT literature makes is that, given
enough data, a Transformer can *learn* better assumptions than we can hand-code, and overtake. The
question this project asks is: **where does that crossover happen?** We answer it by training both
families on the same 32×32 images at three data scales.

## The result

| dataset | images / classes | best CNN | best ViT | winner |
|---|---|---|---|---|
| CIFAR-10 | 50k / 10 | **92.7%** | 85.1% | CNN |
| CIFAR-100 | 50k / 100 | **74.3%** | 62.6% | CNN |
| ImageNet-32 | 1.28M / 1000 | 41.7% | **43.0%** | ViT |

*(top-1 accuracy; the CNN is ResNet-18, the ViT a parameter-matched 11M model)*

The winner flips. On the two smaller datasets the CNN's built-in locality is decisive; on ImageNet-32 —
25× more data at the same image size — the ViT's learned representation finally wins. The absolute
numbers aren't comparable across rows (1000 classes is a far harder problem than 10), but the *sign of
the gap* is, and it changes.

For scale: the paper that introduced ImageNet-32 (Chrabaszcz et al., 2017) reports 59.0% top-1 for a
much larger network trained much longer. Our 41–43% from an 11M-parameter model in 40 epochs is a sane
number in that light — random guessing on 1000 classes is 0.1%.

## Did we rig the race?

A 1.3-point win only means something if the two models were treated fairly, so we ran controls.

The ViT needs a different recipe than the CNN, heavier data augmentation most of all. Given only the
CNN's light crop-and-flip, our first ImageNet ViT simply *memorized* the training set: 97% train accuracy
against 33% validation, on 1.28 million images. So the fair question isn't "same recipe," it's "did the
ViT's win come from any single advantage the CNN didn't get?" Three controls answer it:

- **The ViT with no epoch advantage** (the CNN's 40 epochs) still scores 42.3%, above the best CNN. The
  win isn't bought with training time.
- **The CNN given the ViT's heavy augmentation** drops to 33.9% — mixup makes the CNN *worse*.
- **Bigger models don't close it:** ResNet-50 (2× the parameters) and ViT-base (8×) both land near the
  smaller models. Within our compute budget, more parameters bought nothing.

The middle control is the interesting one, and it explains why the recipes differ. Heavy augmentation
isn't a bonus the CNN would want — it hurts the CNN. It's a substitute for the inductive bias the ViT
lacks: the CNN is already regularized by its own structure (weight sharing and locality physically limit
what it can memorize), while the ViT has nothing holding it back, so the augmentation has to. The two
families need different recipes because they have different problems.

One earlier wrinkle, now fixed: gradient clipping, added to stabilize ResNet-50, quietly cost ResNet-18
about five points. The CNN numbers above use the un-clipped run, and clipping now defaults off for the
models that don't need it.

## What we're comparing

All four models see identical data and are adapted for 32×32 the same way. Torchvision's stock models
target 224×224 and would destroy a tiny image, so the ResNets get a 3×3 stride-1 stem with no early
max-pool (otherwise a 32×32 image is crushed to 8×8 before any real work), and the ViTs use a patch size
of 4 for an 8×8 grid of 64 tokens (the default of 16 would give four tokens for the whole image).

| | resnet18 | resnet50 | vit | vit_base |
|---|---|---|---|---|
| family | CNN | CNN | Transformer | Transformer |
| parameters | 11.7M | 25.5M | 11.1M | 85.9M |
| its job | the CNN baseline | is it just "bigger CNN wins"? | the crossover | does scaling the ViT help? |

`resnet18` (11.7M) and `vit` (11.1M) are parameter-matched on purpose: that's the fair fight, and neither
can win just by being larger.

The recipes differ by family, because each needs something different:

| | CNN | Transformer |
|---|---|---|
| optimizer | SGD + Nesterov momentum | AdamW |
| learning rate | 0.2 (scaled as 0.1 × batch/256) | 0.001 |
| augmentation | crop + flip | crop + flip + mixup + CutMix + erasing |

Everything else is shared: batch 512, a 5-epoch warmup into cosine decay, label smoothing, mixed
precision. Two things worth knowing if you read the code: a Transformer's warmup isn't optional (without
it the attention softmax saturates on the first noisy batches and never recovers), and AdamW's weight
decay isn't comparable to SGD's — AdamW decouples decay from the gradient, so the same-looking number
means something different.

## Engineering notes — what surprised us

The study was only practical because the training pipeline is fast, and making it fast taught us a few
things worth writing down.

- **The whole dataset lives in GPU memory.** ImageNet-32 as raw `uint8` is 3.9 GB and the card has 96 GB,
  so we upload it once and generate every augmented batch on the GPU: no DataLoader, no worker processes,
  no per-batch copy from the host. This only works because the images are tiny — the real 224×224 ImageNet
  would be about 190 GB.
- **Find out which side is waiting before optimizing.** A single CPU core augments about 4,000 images a
  second; the GPU trains at about 13,000. Low GPU utilization during training means you're CPU-bound, and
  no amount of GPU tuning helps until you move the work off the CPU.
- **`channels_last` is slower at 32×32, not faster.** The usual "free speedup for convnets" needs
  ImageNet-sized images to pay off; at CIFAR scale it's pure overhead. Measure before copying advice.
- **A learning-rate error is the difference between training and not training.** We briefly scaled the LR
  from the wrong baseline and got double the correct value: ResNet-18's accuracy peaked at epoch 2 and
  fell, and ResNet-50 diverged to NaN on epoch 1. The trainer now aborts immediately on a NaN loss.
- **On a GPU, a Python loop over samples is the enemy.** Our first random-erasing augmentation looped over
  the fraction of the batch being erased and cut ViT throughput by 3×; rewriting it as one broadcast mask
  restored it. A few hundred tiny kernel launches per batch is all overhead.
- **A second GPU buys a second experiment, not a faster one.** The cards are power-limited, so two jobs on
  one card just split its throughput. The second card's real use is to run a *different* model — which is
  what `train_fleet.py` does. (Splitting one model across both cards measured 0.98× here.)
- **Augmented-train accuracy isn't comparable to clean-test accuracy.** They're different exams; with heavy
  augmentation the training number can even fall below the test number. Don't read overfitting off the
  training printout — this one bit us more than once.

## Status and honest caveats

The training is complete and all eight models are published (see [`a1-cv/README.md`](a1-cv/README.md)).

- **Nothing here is fully converged.** Most runs were still improving when their schedule ended; longer
  schedules would raise every number, and would help `vit_base` most (86M parameters in 40 epochs is
  undertrained).
- **One seed per configuration.** We haven't measured run-to-run variance, so the ViT's ~1-point margin on
  ImageNet-32 is suggestive, not decisive.
- We report accuracy on the official 50k validation split; ImageNet's test split has no public labels.

## Where to look

- **`a1-cv/report_crossover.ipynb`** — the results as a runnable notebook: the crossover, the
  accuracy-versus-cost tradeoff, and how to load any model.
- **`a1-cv/report_factory_performance.ipynb`** — the engineering journey, from the coursework training
  loop to this fast pipeline, with the measurements behind the notes above.
- **`a1-cv/imagenet_data.py`** — the GPU-resident dataset, the most unusual piece of the code.
- **`a1-cv/README.md`** — how to set up, load a model, and run the training.
