"""
data.py -- the whole ImageNet-32 dataset, resident in GPU memory, handing back augmented batches.

THE IDEA
--------
Back on CIFAR-10 we measured the thing that actually hurts: a single CPU core augments ~4,000 img/s
while the card trains at ~13,000 img/s, so the CPU pipeline was starving the GPU by 3x and the card
spent most of its life waiting on Python.  The textbook fix is "throw more DataLoader workers at it."
Here we can do better than that -- we can delete the problem outright.

Here is the trick.  ImageNet-32's entire training set, as raw uint8, is:

    1,281,167 x 32 x 32 x 3 = 3.9 GB

and the RTX PRO 6000 has 96 GB.  So we upload the ENTIRE DATASET to the GPU exactly once, at startup,
and then generate every batch on the card.  Augmentation (random crop + horizontal flip) stops being
a pile of PIL calls on the CPU and becomes a handful of tensor ops on data that already lives in VRAM.
What that buys us:

  * no DataLoader, no worker processes, no 'spawn' problems, no persistent_workers
  * no PIL, no JPEG decode, no Python in the inner loop
  * no host->device copy per batch -- the data never leaves the GPU
  * the CPU is left completely free

WHY DOES THIS EVEN WORK?
------------------------
It's the same trick as the GPU-resident hyperparameter sweep back in CSED 503 A1, just pointed at
images instead of scalars.  And it works ONLY because the images are tiny: you could never do this
with 224x224 ImageNet, which would be ~190 GB and blow straight past the card.  32x32 is the whole
reason the dataset fits -- that single fact is what makes this entire pipeline possible.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')


def load_stats() -> dict:
    """Read stats.json (per-channel mean/std + class count) that prepare_data.py computed for us.

    We always load THESE numbers rather than hardcoding CIFAR's: they were measured from this dataset's
    own pixels, and normalizing ImageNet-32 with CIFAR's constants would quietly skew every image.
    """
    p = os.path.join(DATA_DIR, 'stats.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} not found -- run prepare_data.py first.')
    return json.load(open(p))


class GpuImageNet32:
    """The whole dataset, resident in GPU memory, serving augmented batches.

    This is the one class that stands in for torchvision's transforms + Dataset + DataLoader -- the
    entire stack collapses into ~30 lines of tensor code, because once the pixels already live on the
    GPU there is simply nothing left for a DataLoader to do.
    """

    def __init__(self, device: torch.device, split: str = 'train', subset: int | None = None,
                 seed: int = 0):
        # Step 1: Open the arrays as memmaps rather than reading them whole.  For a subset run the
        # random pick in Step 2 then pulls just the rows we keep off disk, instead of materializing all
        # 3.9 GB in host RAM only to throw most of it away.
        stats = load_stats()
        x = np.load(os.path.join(DATA_DIR, f'{split}_x.npy'), mmap_mode='r')
        y = np.load(os.path.join(DATA_DIR, f'{split}_y.npy'), mmap_mode='r')

        # Step 2: If we only want a subset (smoke tests), sample it at RANDOM across the whole file.
        # CRITICAL: the file is sorted by class, so a SEQUENTIAL subset would contain only a handful of
        # classes -- the first 10,000 rows hold just 8 of the 1000!  A --smoke-test that trained on 8
        # classes and then reported 95% accuracy would be a genuinely evil bug: green, and lying.
        if subset is not None and subset < len(x):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(x), size=subset, replace=False))
            x, y = x[idx], y[idx]

        # Step 3: Upload the whole thing to the card, ONCE, and keep the pixels as uint8 on purpose.
        # (N, 32, 32, 3) uint8 is 3.9 GB, versus 15.7 GB if we stored it as float32 -- so we hold the
        # cheap form and do the uint8 -> float conversion per-batch in _to_float(), which on the GPU is
        # nearly free.  That trade (a little compute per batch to save 12 GB of VRAM) is the whole game.
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).to(device, non_blocking=True).long()
        self.device = device
        self.n = len(self.y)

        # Step 4: Stash the normalization constants shaped as (1,3,1,1) so a single subtract/divide
        # broadcasts cleanly over a whole batch -- (1,3,1,1) vs (B,3,H,W) -> (B,3,H,W), one op, no loop.
        self.mean = torch.tensor(stats['mean'], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(stats['std'], device=device).view(1, 3, 1, 1)
        self.n_classes = stats['n_classes']

    def gb(self) -> float:
        """VRAM this split occupies: uint8 pixels (1 byte) + int64 labels (8 bytes), in GB -- for the log."""
        return (self.x.numel() + self.y.numel() * 8) / 1e9

    def _to_float(self, xb: torch.Tensor) -> torch.Tensor:
        """One batch of raw bytes -> a normalized float batch: (B,32,32,3) uint8 -> (B,3,32,32) float.

        This is EXACTLY torchvision's ToTensor + Normalize, done as two in-place tensor ops on the GPU
        instead of per-image on the CPU.  Doing it per-batch (not once, up front) is precisely what lets
        us keep the resident dataset as uint8 in the first place.
        """
        xb = xb.permute(0, 3, 1, 2).float().div_(255.0)      # (B,32,32,3) NHWC uint8 -> (B,3,32,32) NCHW float in [0,1]
        return xb.sub_(self.mean).div_(self.std)             # ELEMENT-WISE (x - mean) / std, (1,3,1,1) broadcast over (B,3,H,W)

    def _augment(self, xb: torch.Tensor) -> torch.Tensor:
        """RandomCrop(32, padding=4) + RandomHorizontalFlip, batched, on the GPU.

        These are EXACTLY the two transforms from the CIFAR notebook -- we haven't changed the
        augmentation, only WHERE it runs.  Instead of PIL calls on one image at a time, each transform
        becomes a few tensor ops on the whole batch at once.  This method is the payoff of keeping the
        data on the card: no per-image Python, no host round-trip.
        """
        # Variables:
        #   b = batch size (how many images we're augmenting this call)
        #   d = the device the batch lives on (some cuda:N).  We pass device=d to EVERY new tensor so
        #       nothing we allocate accidentally lands on the CPU and forces a copy back and forth.
        b, d = xb.shape[0], xb.device

        # Step 1 -- RandomCrop(32, padding=4): pad each image out to 40x40, then cut a random 32x32
        # window back out of it.  A "random crop" is nothing but padding + a random offset.  We build
        # the window with broadcast index arithmetic so the WHOLE batch is cropped in one indexing op.
        xp = F.pad(xb, (4, 4, 4, 4))                          # (B,3,40,40), zero padding
        i = torch.randint(0, 9, (b,), device=d)               # 9 = 40 - 32 + 1 valid top offsets, one per image
        j = torch.randint(0, 9, (b,), device=d)               # ditto, the left offset
        ar = torch.arange(32, device=d)                       # (32,) the 32 pixel positions inside a window
        rows = (i.view(b, 1, 1) + ar.view(1, 32, 1))          # (B,1,1) + (1,32,1) -> (B,32,1) row index per image
        cols = (j.view(b, 1, 1) + ar.view(1, 1, 32))          # (B,1,1) + (1,1,32) -> (B,1,32) col index per image
        bidx = torch.arange(b, device=d).view(b, 1, 1)        # (B,1,1) which image each element belongs to
        # GOTCHA: with a slice (the `:`) sitting BETWEEN the index tensors, PyTorch's advanced indexing
        # pulls the indexed dims to the front, so xp[bidx, :, rows, cols] comes back as (B,32,32,C), NOT
        # (B,C,32,32).  That's the whole reason we permute(0,3,1,2) right after -- to put channels back.
        xb = xp[bidx, :, rows, cols].permute(0, 3, 1, 2).contiguous()

        # Step 2 -- RandomHorizontalFlip(): mirror a random half of the batch.
        # WHY torch.where AND NOT A MASKED WRITE?
        # ---------------------------------------
        # The obvious "xb[flip] = xb[flip].flip(-1)" runs nonzero() under the hood, and nonzero() forces
        # a GPU->host sync every single time -- once per batch, on every one of the ~2,500 batches/epoch.
        # torch.where computes the same result branchlessly with no sync: same RNG draw, identical
        # distribution of flipped images, just none of the stalls.
        flip = (torch.rand(b, device=d) < 0.5).view(b, 1, 1, 1)   # (B,1,1,1) True for the half we mirror
        xb = torch.where(flip, xb.flip(-1), xb)               # (B,1,1,1) vs (B,C,H,W) -> (B,C,H,W), ELEMENT-WISE pick
        return xb

    def epoch(self, batch_size: int, train: bool, generator: torch.Generator | None = None):
        """Yield (images, labels) for one pass over the split.

        This is the DataLoader-shaped surface the training loop iterates.  When train=True we shuffle the
        order and augment each batch; when False (validation) we walk straight through in order with
        neither.  Everything -- shuffle, slice, augment -- happens on the GPU, so a batch is never copied
        from host to device on its way out.
        """
        # Step 1: Decide the visiting order.  Training shuffles (through the passed-in generator, so a
        # run stays reproducible); eval just walks the split in its natural order, where order is moot.
        if train:
            order = torch.randperm(self.n, device=self.device, generator=generator)
        else:
            order = torch.arange(self.n, device=self.device)

        # Step 2: Walk that order in batch_size chunks.  Each iteration gathers this chunk's rows ->
        # normalizes them to float -> (training only) augments -> hands back (images, labels).  The last
        # chunk is simply whatever's left over; we keep it rather than dropping a partial batch.
        for s in range(0, self.n, batch_size):
            idx = order[s:s + batch_size]
            xb = self._to_float(self.x[idx])
            if train:
                xb = self._augment(xb)
            yield xb, self.y[idx]

    def n_batches(self, batch_size: int) -> int:
        """How many batches one epoch yields -- ceil(n / batch_size), so the partial last batch counts."""
        return (self.n + batch_size - 1) // batch_size


# ===================================================================================================
# STRONG AUGMENTATION FOR THE TRANSFORMERS -- again all on the GPU, again pure tensor math.
#
# WHY DO ONLY THE ViTs NEED THIS?
# -------------------------------
# Our very first ImageNet-32 run fed the ViTs only crop+flip, the same diet as the CNN, and they
# MEMORIZED the dataset: 97.4% train accuracy against 32.6% validation, a 65-point gap, on 1.28M
# images.  The ResNet, on that identical augmentation, sat at a healthy +8.7% gap.  The gap between
# those gaps IS the inductive bias, made visible.
#
# Here is the mechanism.  A CNN CAN'T easily memorize -- weight sharing and locality physically
# restrict what it is able to represent, so the architecture regularizes itself for free.  A ViT has
# no such constraint: every token can attend to every other, so it has more than enough capacity to
# simply learn the training set by heart.  Augmentation is the SUBSTITUTE for the prior the transformer
# doesn't carry.  That's why every serious ViT recipe (DeiT and its successors) ships mixup + CutMix +
# erasing, and why a CNN recipe can get away without any of them.
# ===================================================================================================

def random_erasing_(x: torch.Tensor, p: float = 0.25, scale=(0.02, 0.2)) -> torch.Tensor:
    """Erase a random rectangle from a random subset of the batch, in place.  GPU-side RandomErasing.

    The trailing `_` follows the PyTorch convention for "mutates its argument" -- we overwrite x rather
    than allocate a copy.

    FULLY VECTORIZED -- and it HAS to be.  The first version looped in Python over the ~25% of the batch
    that actually gets erased and did one slice-assign per image; that dropped ViT throughput from
    14.3k to 4.2k img/s (a 3.4x hit!), because a few hundred tiny GPU ops per batch is nothing but
    kernel-launch overhead.  Building the erase mask out of broadcast comparisons instead collapses the
    whole thing to ~5 kernels for the entire batch, no matter how big the batch is.  Same lesson as the
    CIFAR launch-bound analysis: on a GPU, a Python loop over samples is the enemy. :-)
    """
    # Variables:
    #   b       = batch size;  h, w = image height / width
    #   p       = per-image probability that this image gets a rectangle erased at all
    #   eh, ew  = erase-rectangle height / width, (B,), one per image
    #   top,left= top-left corner of each image's rectangle, (B,)
    b, _, h, w = x.shape
    d = x.device

    # Step 1 -- pick a random rectangle SIZE for every image, all at once.
    #   Part A: sample a target area as a fraction (scale) of the h*w total.
    #   Part B: sample an aspect ratio, then split that area into a height and a width.
    # We clamp to [1, h-1] / [1, w-1] so the box is always at least 1px and never the entire image.
    area = h * w * (torch.rand(b, device=d) * (scale[1] - scale[0]) + scale[0])
    ratio = torch.empty(b, device=d).uniform_(0.3, 3.3)
    eh = (area * ratio).sqrt().clamp(1, h - 1).long()          # (B,) rectangle heights
    ew = (area / ratio).sqrt().clamp(1, w - 1).long()          # (B,) rectangle widths

    # Step 2 -- pick WHERE each rectangle goes: a random top-left that keeps the box fully inside.
    top = (torch.rand(b, device=d) * (h - eh).float()).long()  # (B,) each in [0, h-eh]
    left = (torch.rand(b, device=d) * (w - ew).float()).long() # (B,) each in [0, w-ew]

    # Step 3 -- turn all those per-image boxes into ONE boolean mask, with pure broadcasting.
    # Equivalent loop: for image k, mask[k] = (top[k] <= row < top[k]+eh[k]) AND the same test on cols.
    # We do it for every image at once by broadcasting the row/col ranges against the per-image bounds:
    # (1,H,1) vs (B,1,1) -> (B,H,1), then AND'd with the column half -> (B,H,W).
    rows = torch.arange(h, device=d).view(1, h, 1)             # (1,H,1), broadcast against (B,1,1)
    cols = torch.arange(w, device=d).view(1, 1, w)             # (1,1,W), broadcast against (B,1,1)
    inside = ((rows >= top.view(b, 1, 1)) & (rows < (top + eh).view(b, 1, 1)) &
              (cols >= left.view(b, 1, 1)) & (cols < (left + ew).view(b, 1, 1)))
    hit = (torch.rand(b, device=d) < p).view(b, 1, 1)          # (B,1,1) which images get erased at all
    mask = (inside & hit).unsqueeze(1)                          # (B,H,W) -> (B,1,H,W), broadcasts over C
    # Step 4 -- write the erased value, and here's the subtlety: the batch is ALREADY normalized, so
    # 0.0 is the per-channel MEAN, not black.  Erasing to the mean is exactly what torchvision's
    # RandomErasing does post-normalize -- it hides the region without inventing a hard edge.
    return x.masked_fill_(mask, 0.0)                           # 0 == the channel mean, post-normalize


def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0, prob: float = 0.5):
    """Blend each image with ANOTHER image from the same batch.  Returns (x, y_a, y_b, lam).

    Two flavors, and on any given batch we roll a die (prob) for which one to apply:
      * mixup:  a weighted pixel average of two images; the target becomes that same weighted mix.
      * CutMix: paste a rectangular patch of image B into image A; the target mixes by the AREA of the
        patch that got pasted in.

    Either way the trainer's loss becomes  lam * CE(out, y_a) + (1 - lam) * CE(out, y_b)  -- i.e. the
    model is NEVER shown a confident one-hot target, so it can't simply memorize image->label pairs,
    which is exactly the crutch the ViT reached for without it (see the big note above).  We hand back
    y_a, y_b and lam rather than a soft label so the caller can build that two-term loss itself.
    """
    # Variables:
    #   b    = batch size
    #   perm = a random permutation of [0..b) -- image i gets blended with image perm[i].  Pairing each
    #          image with another slot of the SAME batch is what makes the partner free and copy-less.
    #   lam  = the mixing weight in [0,1]: how much of image A survives (1.0 == no mixing at all)
    b = x.size(0)
    perm = torch.randperm(b, device=x.device)

    # Roll once: with probability `prob` we do mixup, otherwise CutMix.
    if torch.rand(()) < prob:                                  # ---- mixup: whole-image weighted average
        # Beta(0.2, 0.2) is U-shaped -> lam lands near 0 or 1 most of the time, so a blended image still
        # mostly resembles one of its two parents rather than a muddy 50/50 ghost.
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        x = lam * x + (1 - lam) * x[perm]
    else:                                                       # ---- CutMix: paste a patch of B into A
        # Beta(1.0, 1.0) is just Uniform(0,1); here lam sets the SIZE of the pasted rectangle.
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        h, w = x.shape[2:]
        # A box whose area fraction is (1 - lam) -> its side lengths scale as sqrt(1 - lam).
        rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        if rh > 0 and rw > 0:
            # Random top-left, then overwrite that window of every image A with the same window of B.
            cy = int(torch.randint(0, h - rh + 1, ()))
            cx = int(torch.randint(0, w - rw + 1, ()))
            x[:, :, cy:cy + rh, cx:cx + rw] = x[perm][:, :, cy:cy + rh, cx:cx + rw]
            # IMPORTANT: recompute lam from the ACTUAL pasted area, not the sampled value.  Integer
            # rounding of rh/rw means the real patch rarely matches the drawn lam exactly, and the loss
            # weight has to match what we TRULY pasted or the target is subtly wrong.
            lam = 1 - (rh * rw) / (h * w)                       # true mixed area, not the sampled lam
        else:
            # The box rounded away to nothing (lam ~ 1): no paste happened, so it stays a clean image.
            lam = 1.0
    return x, y, y[perm], lam
