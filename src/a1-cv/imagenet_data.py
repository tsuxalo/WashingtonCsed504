"""
The whole ImageNet-32 dataset, resident in GPU memory, handing back augmented batches.

Back on CIFAR-10 we measured the thing that actually hurts: a single CPU core augments about 4,000
img/s while the card trains at about 13,000 img/s, so the CPU pipeline was starving the GPU by 3x
and the card spent most of its life waiting on Python. The textbook fix is to throw more DataLoader
workers at it. Here we can do better than that and delete the problem outright.

Here is the trick. ImageNet-32's entire training set, as raw uint8, is:

    1,281,167 x 32 x 32 x 3 = 3.9 GB

and the RTX PRO 6000 has 96 GB. So we upload the entire dataset to the GPU exactly once, at startup,
and then generate every batch on the card. Augmentation (random crop plus horizontal flip) stops
being a pile of PIL calls on the CPU and becomes a handful of tensor ops on data that already lives
in VRAM. What that buys us:

  - no DataLoader, no worker processes, no spawn problems, no persistent_workers
  - no PIL, no JPEG decode, no Python in the inner loop
  - no host-to-device copy per batch, since the data never leaves the GPU
  - the CPU is left completely free

It is the same trick as the GPU-resident hyperparameter sweep back in CSED 503 A1, just pointed at
images instead of scalars. And it works only because the images are tiny: you could never do this
with 224x224 ImageNet, which would be about 190 GB and blow straight past the card. 32x32 is the
whole reason the dataset fits, and that single fact is what makes this entire pipeline possible.
"""
from __future__ import annotations

import json
import os

import numpy as np
import torch
import torch.nn.functional as F

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data', 'imagenet32')


def load_stats() -> dict:
    """Read stats.json (per-channel mean/std plus class count) that imagenet_prepare.py computed for us.

    We always load these numbers rather than hardcoding CIFAR's: they were measured from this dataset's
    own pixels, and normalizing ImageNet-32 with CIFAR's constants would quietly skew every image.
    """
    p = os.path.join(DATA_DIR, 'stats.json')
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} not found -- run imagenet_prepare.py first.')
    return json.load(open(p))


class GpuImageNet32:
    """The whole dataset, resident in GPU memory, serving augmented batches.

    This is the one class that stands in for torchvision's transforms, Dataset, and DataLoader. The
    entire stack collapses into about 30 lines of tensor code, because once the pixels already live on
    the GPU there is simply nothing left for a DataLoader to do.
    """

    def __init__(self, device: torch.device, split: str = 'train', subset: int | None = None,
                 seed: int = 0):
        # Step 1: Open the arrays as memmaps rather than reading them whole. For a subset run, the
        # random pick in Step 2 then pulls just the rows we keep off disk, instead of materializing
        # all 3.9 GB in host RAM only to throw most of it away.
        stats = load_stats()
        x = np.load(os.path.join(DATA_DIR, f'{split}_x.npy'), mmap_mode='r')
        y = np.load(os.path.join(DATA_DIR, f'{split}_y.npy'), mmap_mode='r')


        # Step 2: If we only want a subset (smoke tests), sample it at random across the whole file.
        # The file is sorted by class, so a sequential subset would contain only a handful of classes:
        # the first 10,000 rows hold just 8 of the 1000. A --smoke-test that trained on 8 classes and
        # then reported 95% accuracy would be a genuinely evil bug, green and lying.
        if subset is not None and subset < len(x):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(x), size=subset, replace=False))
            x, y = x[idx], y[idx]


        # Step 3: Upload the whole thing to the card once, and keep the pixels as uint8 on purpose.
        #
        # x: (N, 32, 32, 3) uint8
        #
        # At (N, 32, 32, 3) uint8 the dataset is 3.9 GB, versus 15.7 GB if we stored it as float32, so
        # we hold the cheap form and do the uint8 to float conversion per batch in _to_float(), which
        # on the GPU is nearly free. That trade, a little compute per batch to save 12 GB of VRAM, is
        # the whole game.

        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(device, non_blocking=True)
        self.y = torch.from_numpy(np.ascontiguousarray(y)).to(device, non_blocking=True).long()
        self.device = device
        self.n = len(self.y)


        # Step 4: Stash the normalization constants shaped as (1, 3, 1, 1) so a single subtract and
        # divide broadcasts cleanly over a whole batch.
        #
        # mean: (1, 3, 1, 1)
        # std:  (1, 3, 1, 1)
        #
        # Broadcasting (1, 3, 1, 1) against a batch of (B, 3, H, W) gives (B, 3, H, W) in one op, with
        # no loop.

        self.mean = torch.tensor(stats['mean'], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(stats['std'], device=device).view(1, 3, 1, 1)
        self.n_classes = stats['n_classes']

    def gb(self) -> float:
        """VRAM this split occupies, in GB: uint8 pixels (1 byte) plus int64 labels (8 bytes), for the log."""
        return (self.x.numel() + self.y.numel() * 8) / 1e9

    def _to_float(self, xb: torch.Tensor) -> torch.Tensor:
        """Turn one batch of raw bytes into a normalized float batch.

        Inputs:
         - xb: raw image batch, of shape (B, 32, 32, 3) uint8

        Returns:
         - a normalized float batch, of shape (B, 3, 32, 32)

        This is exactly torchvision's ToTensor plus Normalize, done as two in-place tensor ops on the
        GPU instead of per-image on the CPU. Doing it per batch, not once up front, is precisely what
        lets us keep the resident dataset as uint8 in the first place.
        """
        # Reorder NHWC to NCHW and scale into [0, 1].
        #
        # xb (before): (B, 32, 32, 3) NHWC uint8
        # xb (after):  (B, 3, 32, 32) NCHW float in [0, 1]

        xb = xb.permute(0, 3, 1, 2).float().div_(255.0)

        # Normalize elementwise as (x - mean) / std, with the (1, 3, 1, 1) constants broadcasting over
        # the (B, 3, H, W) batch.

        return xb.sub_(self.mean).div_(self.std)

    def _augment(self, xb: torch.Tensor) -> torch.Tensor:
        """RandomCrop(32, padding=4) and RandomHorizontalFlip, batched, on the GPU.

        These are exactly the two transforms from the CIFAR notebook. We haven't changed the
        augmentation, only where it runs. Instead of PIL calls on one image at a time, each transform
        becomes a few tensor ops on the whole batch at once. This method is the payoff of keeping the
        data on the card: no per-image Python, no host round-trip.
        """
        # b is the batch size, how many images we're augmenting this call. d is the device the batch
        # lives on (some cuda:N). We pass device=d to every new tensor so nothing we allocate
        # accidentally lands on the CPU and forces a copy back and forth.
        b, d = xb.shape[0], xb.device


        # Step 1: RandomCrop(32, padding=4). Pad each image out to 40x40, then cut a random 32x32
        # window back out of it. A random crop is nothing but padding plus a random offset. We build
        # the window with broadcast index arithmetic so the whole batch is cropped in one indexing op.
        #
        # xp:   (B, 3, 40, 40) zero padding
        # i:    (B,) valid top offsets, one per image, since 9 = 40 - 32 + 1
        # j:    (B,) the matching left offset
        # ar:   (32,) the 32 pixel positions inside a window
        # rows: (B, 32, 1) row index per image, from (B, 1, 1) plus (1, 32, 1)
        # cols: (B, 1, 32) col index per image, from (B, 1, 1) plus (1, 1, 32)
        # bidx: (B, 1, 1) which image each element belongs to

        xp = F.pad(xb, (4, 4, 4, 4))
        i = torch.randint(0, 9, (b,), device=d)
        j = torch.randint(0, 9, (b,), device=d)
        ar = torch.arange(32, device=d)
        rows = (i.view(b, 1, 1) + ar.view(1, 32, 1))
        cols = (j.view(b, 1, 1) + ar.view(1, 1, 32))
        bidx = torch.arange(b, device=d).view(b, 1, 1)

        # With a slice (the `:`) sitting between the index tensors, PyTorch's advanced indexing pulls
        # the indexed dims to the front, so xp[bidx, :, rows, cols] comes back as (B, 32, 32, C), not
        # (B, C, 32, 32). That is the whole reason we permute(0, 3, 1, 2) right after, to put the
        # channels back.
        #
        # xb: (B, 3, 32, 32)

        xb = xp[bidx, :, rows, cols].permute(0, 3, 1, 2).contiguous()


        # Step 2: RandomHorizontalFlip(). Mirror a random half of the batch.
        #
        # flip: (B, 1, 1, 1) True for the half we mirror
        # xb:   (B, C, H, W) after the elementwise pick between the flipped and original batch
        #
        # We use torch.where rather than a masked write. The obvious "xb[flip] = xb[flip].flip(-1)"
        # runs nonzero() under the hood, and nonzero() forces a GPU-to-host sync every single time,
        # once per batch, on every one of the about 2,500 batches per epoch. torch.where computes the
        # same result branchlessly with no sync: the same RNG draw, an identical distribution of
        # flipped images, just none of the stalls.

        flip = (torch.rand(b, device=d) < 0.5).view(b, 1, 1, 1)
        xb = torch.where(flip, xb.flip(-1), xb)
        return xb

    def epoch(self, batch_size: int, train: bool, generator: torch.Generator | None = None):
        """Yield (images, labels) for one pass over the split.

        This is the DataLoader-shaped surface the training loop iterates. When train=True we shuffle
        the order and augment each batch; when False (validation) we walk straight through in order
        with neither. Everything, shuffle and slice and augment, happens on the GPU, so a batch is
        never copied from host to device on its way out.
        """
        # Step 1: Decide the visiting order. Training shuffles (through the passed-in generator, so a
        # run stays reproducible); eval just walks the split in its natural order, where order is moot.
        if train:
            order = torch.randperm(self.n, device=self.device, generator=generator)
        else:
            order = torch.arange(self.n, device=self.device)


        # Step 2: Walk that order in batch_size chunks. Each iteration gathers this chunk's rows,
        # normalizes them to float, augments them when training, and hands back (images, labels). The
        # last chunk is simply whatever is left over; we keep it rather than dropping a partial batch.
        for s in range(0, self.n, batch_size):
            idx = order[s:s + batch_size]
            xb = self._to_float(self.x[idx])
            if train:
                xb = self._augment(xb)
            yield xb, self.y[idx]

    def n_batches(self, batch_size: int) -> int:
        """How many batches one epoch yields: ceil(n / batch_size), so the partial last batch counts."""
        return (self.n + batch_size - 1) // batch_size


# Strong augmentation for the transformers, again all on the GPU, again pure tensor math.
#
# Our very first ImageNet-32 run fed the ViTs only crop and flip, the same diet as the CNN, and they
# memorized the dataset: 97.4% train accuracy against 32.6% validation, a 65-point gap, on 1.28M
# images. The ResNet, on that identical augmentation, sat at a healthy +8.7% gap. The gap between
# those gaps is the inductive bias, made visible.
#
# Here is the mechanism. A CNN can't easily memorize: weight sharing and locality physically restrict
# what it is able to represent, so the architecture regularizes itself for free. A ViT has no such
# constraint, since every token can attend to every other, so it has more than enough capacity to
# simply learn the training set by heart. Augmentation is the substitute for the prior the transformer
# doesn't carry. That is why every serious ViT recipe (DeiT and its successors) ships mixup plus
# CutMix plus erasing, and why a CNN recipe can get away without any of them.
def random_erasing_(x: torch.Tensor, p: float = 0.25, scale=(0.02, 0.2)) -> torch.Tensor:
    """Erase a random rectangle from a random subset of the batch, in place. GPU-side RandomErasing.

    The trailing underscore follows the PyTorch convention for a method that mutates its argument: we
    overwrite x rather than allocate a copy.

    This is fully vectorized, and it has to be. The first version looped in Python over the roughly
    25% of the batch that actually gets erased and did one slice-assign per image; that dropped ViT
    throughput from 14.3k to 4.2k img/s, a 3.4x hit, because a few hundred tiny GPU ops per batch is
    nothing but kernel-launch overhead. Building the erase mask out of broadcast comparisons instead
    collapses the whole thing to about 5 kernels for the entire batch, no matter how big the batch is.
    Same lesson as the CIFAR launch-bound analysis: on a GPU, a Python loop over samples is the enemy.
    """
    # b is the batch size, and h, w are the image height and width. p is the per-image probability
    # that an image gets a rectangle erased at all. eh, ew are the erase-rectangle height and width,
    # each (B,), one per image. top and left are the top-left corner of each image's rectangle, each
    # (B,).
    b, _, h, w = x.shape
    d = x.device


    # Step 1: Pick a random rectangle size for every image, all at once. First we sample a target
    # area as a fraction (scale) of the h*w total, then we sample an aspect ratio and split that area
    # into a height and a width. We clamp to [1, h-1] and [1, w-1] so the box is always at least 1px
    # and never the entire image.
    #
    # eh: (B,) rectangle heights
    # ew: (B,) rectangle widths

    area = h * w * (torch.rand(b, device=d) * (scale[1] - scale[0]) + scale[0])
    ratio = torch.empty(b, device=d).uniform_(0.3, 3.3)
    eh = (area * ratio).sqrt().clamp(1, h - 1).long()
    ew = (area / ratio).sqrt().clamp(1, w - 1).long()


    # Step 2: Pick where each rectangle goes, a random top-left that keeps the box fully inside.
    #
    # top:  (B,) each in [0, h-eh]
    # left: (B,) each in [0, w-ew]

    top = (torch.rand(b, device=d) * (h - eh).float()).long()
    left = (torch.rand(b, device=d) * (w - ew).float()).long()


    # Step 3: Turn all those per-image boxes into one boolean mask, with pure broadcasting. The
    # equivalent loop would be, for image k, mask[k] = (top[k] <= row < top[k]+eh[k]) and the same
    # test on cols. We do it for every image at once by broadcasting the row and col ranges against
    # the per-image bounds: (1, H, 1) against (B, 1, 1) gives (B, H, 1), which is then and'd with the
    # column half to give (B, H, W).
    #
    # rows: (1, H, 1) broadcast against (B, 1, 1)
    # cols: (1, 1, W) broadcast against (B, 1, 1)
    # hit:  (B, 1, 1) which images get erased at all
    # mask: (B, 1, H, W) from (B, H, W) unsqueezed, broadcasts over C

    rows = torch.arange(h, device=d).view(1, h, 1)
    cols = torch.arange(w, device=d).view(1, 1, w)
    inside = ((rows >= top.view(b, 1, 1)) & (rows < (top + eh).view(b, 1, 1)) &
              (cols >= left.view(b, 1, 1)) & (cols < (left + ew).view(b, 1, 1)))
    hit = (torch.rand(b, device=d) < p).view(b, 1, 1)
    mask = (inside & hit).unsqueeze(1)


    # Step 4: Write the erased value. Here is the subtlety: the batch is already normalized, so 0.0 is
    # the per-channel mean, not black. Erasing to the mean is exactly what torchvision's RandomErasing
    # does post-normalize, since it hides the region without inventing a hard edge.

    return x.masked_fill_(mask, 0.0)


def mixup_cutmix(x: torch.Tensor, y: torch.Tensor, mixup_alpha: float = 0.2,
                 cutmix_alpha: float = 1.0, prob: float = 0.5):
    """Blend each image with another image from the same batch. Returns (x, y_a, y_b, lam).

    There are two flavors, and on any given batch we roll a die (prob) for which one to apply:
     - mixup: a weighted pixel average of two images; the target becomes that same weighted mix.
     - CutMix: paste a rectangular patch of image B into image A; the target mixes by the area of the
       patch that got pasted in.

    Either way the trainer's loss becomes lam * CE(out, y_a) + (1 - lam) * CE(out, y_b), so the model
    is never shown a confident one-hot target and can't simply memorize image-to-label pairs, which is
    exactly the crutch the ViT reached for without it (see the comment above). We hand back y_a, y_b,
    and lam rather than a soft label so the caller can build that two-term loss itself.
    """
    # b is the batch size. perm is a random permutation of [0..b), so image i gets blended with image
    # perm[i]; pairing each image with another slot of the same batch is what makes the partner free
    # and copy-less. lam is the mixing weight in [0, 1], how much of image A survives (1.0 means no
    # mixing at all).
    b = x.size(0)
    perm = torch.randperm(b, device=x.device)

    # Roll once: with probability `prob` we do mixup, otherwise CutMix. This branch is mixup, a
    # whole-image weighted average.
    if torch.rand(()) < prob:
        # Beta(0.2, 0.2) is U-shaped, so lam lands near 0 or 1 most of the time and a blended image
        # still mostly resembles one of its two parents rather than a muddy 50/50 ghost.
        lam = float(torch.distributions.Beta(mixup_alpha, mixup_alpha).sample())
        x = lam * x + (1 - lam) * x[perm]
    # This branch is CutMix: paste a patch of B into A.
    else:
        # Beta(1.0, 1.0) is just Uniform(0, 1); here lam sets the size of the pasted rectangle.
        lam = float(torch.distributions.Beta(cutmix_alpha, cutmix_alpha).sample())
        h, w = x.shape[2:]

        # A box whose area fraction is (1 - lam) has side lengths that scale as sqrt(1 - lam).
        rh, rw = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        if rh > 0 and rw > 0:
            # Random top-left, then overwrite that window of every image A with the same window of B.
            cy = int(torch.randint(0, h - rh + 1, ()))
            cx = int(torch.randint(0, w - rw + 1, ()))
            x[:, :, cy:cy + rh, cx:cx + rw] = x[perm][:, :, cy:cy + rh, cx:cx + rw]
            # Recompute lam from the actual pasted area, not the sampled value. Integer rounding of
            # rh/rw means the real patch rarely matches the drawn lam exactly, and the loss weight has
            # to match what we truly pasted or the target is subtly wrong. This is the true mixed
            # area, not the sampled lam.
            lam = 1 - (rh * rw) / (h * w)
        else:
            # The box rounded away to nothing (lam ~ 1): no paste happened, so it stays a clean image.
            lam = 1.0
    return x, y, y[perm], lam
