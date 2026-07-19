"""
gpu_data.py -- GPU-resident data + augmentation for small (CIFAR-scale) images.

THE IDEA
--------
A torch DataLoader hands augmentation off to CPU worker processes.  That's the right tool when
decode+augment is the bottleneck (see the workers lesson) -- but for 32x32 images a modern GPU
trains a small CNN so fast that even 8 CPU workers (~14k img/s) can't keep it fed, and at batch
128 the model is *launch-bound* anyway.  So we flip it around: the dataset is tiny, so we hold it
on the device as one uint8 NCHW tensor and do crop/flip/erase/normalize right there, in batches.
No DataLoader, no workers, no host->device copy per batch.

WHY does this pay off?
----------------------
Measured on the RTX PRO 6000: CPU-workers bs128 ~13.8k img/s vs GPU-resident bs512 ~18.5k img/s,
at much higher utilization.  The CPU pipeline simply can't feed the card; the GPU-resident one
keeps it saturated.

WHY is it even possible?
------------------------
The dataset is small enough to just *live* in VRAM: CIFAR-100 train = 50k*32*32*3 = 147 MB as
uint8.  Keep it uint8 (not float) and it stays cheap -- we convert to float per batch, which is
nearly free on the GPU.  This is exactly the trick you can NOT pull on 224x224 ImageNet.

It runs on cuda / mps / cpu unchanged -- it's just tensor ops -- but it only *pays off* on a GPU,
which is precisely when you want it.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def to_device_uint8(hf_split, device, img_key='img', label_key='fine_label'):
    """Decode a HuggingFace image split once, straight onto the device as a uint8 NCHW tensor.

    This is the one-time upload that makes the whole scheme work: we pay the PIL decode + host
    copy exactly once, at startup, and everything after that stays on the device.  Labels ride
    along as an int64 tensor on the same device, so nothing in the training loop touches the host.
    """
    imgs = np.stack([np.asarray(im.convert('RGB'), np.uint8) for im in hf_split[img_key]])
    x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).contiguous()  # (N,H,W,C) -> (N,C,H,W) uint8
    y = torch.from_numpy(np.asarray(hf_split[label_key], np.int64)).to(device)
    return x, y


class GPUImageLoader:
    """A DataLoader-shaped iterable over small images that never leaves the device.

    Yields ``(augmented_float_batch, labels)``, everything on ``images_u8.device``.  The
    augmentation -- random crop with reflect padding, horizontal flip, random erasing -- fires
    only when ``train=True``; normalization always.  ``drop_last`` defaults to ``train`` so every
    training batch shares one fixed shape, which keeps cuDNN's autotuner from re-planning on a
    short tail batch.  Deterministic given ``seed``: it draws from a device Generator, so even the
    RNG lives on the device.
    """

    def __init__(self, images_u8, labels, batch_size, mean, std, *, train,
                 crop_pad=4, hflip=True, erasing=False, erase_p=0.25,
                 drop_last=None, seed=None):
        self.x, self.y = images_u8, labels
        self.bs = batch_size
        self.train = train
        self.crop_pad = crop_pad
        self.hflip = hflip
        self.erasing = erasing
        self.erase_p = erase_p
        self.drop_last = train if drop_last is None else drop_last   # fixed batch shape while training
        dev = images_u8.device

        # mean/std shaped (1,3,1,1) so they broadcast cleanly over an (N,3,H,W) batch.
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)

        # A shared 0..H-1 ramp, reused to build the crop and erase index masks (H == W here).
        H = images_u8.size(-1)
        self._ar = torch.arange(H, device=dev)

        # A *device* Generator: same seed -> same augmentation, and every draw stays on the device,
        # so sampling never forces a host sync.
        self.gen = torch.Generator(device=dev)
        if seed is not None:
            self.gen.manual_seed(seed)

    def __len__(self):
        n = self.x.size(0)
        # drop_last -> floor division (whole batches only); otherwise ceil-div to keep the tail.
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def _random_crop(self, x):                       # x: (B,C,H,W) float
        # Variables: B batch, C channels, H/W height/width, m = number of valid crop offsets.
        #
        # RandomCrop(H, padding=crop_pad) for a WHOLE batch at once.  The trick: reflect-pad the
        # image, then pull a per-sample HxW window back out with advanced indexing -- so every
        # image in the batch gets its OWN random offset in a single gather, with no Python loop.
        B, C, H, W = x.shape

        # Step 1: reflect-pad all four sides.  Reflect (not zero) keeps the border as real image
        # content instead of a black frame the model could learn to key off.
        x = F.pad(x, (self.crop_pad,) * 4, mode='reflect')   # (B,C,H+2p,W+2p)

        # Step 2: draw a random top-left corner per sample.  After padding there are m = 2p+1 valid
        # offsets along each axis (0 .. 2p inclusive).
        m = 2 * self.crop_pad + 1
        oy = torch.randint(0, m, (B,), device=x.device, generator=self.gen)   # (B,) row offsets
        ox = torch.randint(0, m, (B,), device=x.device, generator=self.gen)   # (B,) col offsets

        # Step 3: turn each corner into the H (or W) absolute indices it selects, broadcasting the
        # shared ramp against the per-sample offset:
        #   oy.view(B,1) + arange(H) -> (B,H), then reshaped so rows and cols land on separate axes.
        rows = (oy.view(B, 1) + self._ar).view(B, 1, H, 1)   # (B,1,H,1)
        cols = (ox.view(B, 1) + self._ar).view(B, 1, 1, W)   # (B,1,1,W)

        # Step 4: batch and channel index tensors, shaped to broadcast against rows/cols.
        bidx = torch.arange(B, device=x.device).view(B, 1, 1, 1)   # (B,1,1,1)
        cidx = torch.arange(C, device=x.device).view(1, C, 1, 1)   # (1,C,1,1)

        # Step 5: the gather.  All four index tensors broadcast to (B,C,H,W), so the whole batch is
        # cropped in one kernel.
        #   Equivalent loop: for b in range(B): out[b] = x[b, :, oy[b]:oy[b]+H, ox[b]:ox[b]+W]
        return x[bidx, cidx, rows, cols]             # (B,C,H,W), per-sample crop

    def _random_erase(self, x, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        # Variables: B batch, C channels, H/W height/width; h/w = erase-rectangle size per sample,
        # (top, left) = its corner, `do` = which samples get erased at all.
        #
        # RandomErasing for the whole batch with NO Python loop.  We sample a rectangle per image,
        # then paint the erase region as a boolean mask via broadcast comparisons.  Looping in
        # Python over just the ~erase_p fraction that gets erased is the trap: a few hundred tiny
        # GPU ops per batch is pure launch overhead (the same launch-bound lesson from the module
        # header).  One masked write over the batch is a handful of kernels regardless of B.
        B, C, H, W = x.shape
        dev = x.device

        # Step 1: decide who gets erased (Bernoulli(erase_p) per sample).  Samples with do=False
        # drop out of the mask below and are left untouched.
        do = torch.rand(B, device=dev, generator=self.gen) < self.erase_p   # (B,) bool

        # Step 2: pick a target area (a fraction of H*W) and an aspect ratio per sample.  Ratio is
        # sampled in LOG space so that e.g. 3:1 and 1:3 are equally likely -- a uniform draw on the
        # raw ratio would bias toward wide rectangles.
        area = torch.empty(B, device=dev).uniform_(scale[0], scale[1], generator=self.gen) * (H * W)
        logr = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]),
                                                    generator=self.gen)
        ar = torch.exp(logr)                         # (B,) aspect ratio, back in linear space

        # Step 3: area + ratio -> integer rectangle size, clamped to fit inside the image.
        h = (area * ar).sqrt().round().clamp(1, H).long()   # (B,) rect height
        w = (area / ar).sqrt().round().clamp(1, W).long()   # (B,) rect width

        # Step 4: a random top-left corner that keeps the rectangle in bounds.  (H - h + 1) is the
        # count of valid top rows; scale a uniform [0,1) by it and floor.
        top = (torch.rand(B, device=dev, generator=self.gen) * (H - h + 1).float()).long()   # (B,)
        left = (torch.rand(B, device=dev, generator=self.gen) * (W - w + 1).float()).long()  # (B,)

        # Step 5: build the erase mask.  Compare the shared row/col ramp against each sample's
        # [top, top+h) x [left, left+w) box, then AND in `do` to keep only the chosen samples.
        #   rows (1,1,H,1) & cols (1,1,1,W) vs (B,1,1,1) corners -> mask (B,1,H,W), broadcasts over C.
        rows = self._ar.view(1, 1, H, 1)             # (1,1,H,1)
        cols = self._ar.view(1, 1, 1, W)             # (1,1,1,W)
        mask = ((rows >= top.view(B, 1, 1, 1)) & (rows < (top + h).view(B, 1, 1, 1)) &
                (cols >= left.view(B, 1, 1, 1)) & (cols < (left + w).view(B, 1, 1, 1)) &
                do.view(B, 1, 1, 1))

        # Step 6: ELEMENT-WISE select 0 where the mask is True, x elsewhere.  torch.where (not an
        # in-place masked write) keeps this a pure element-wise kernel with NO host sync.  0 is the
        # post-normalize channel mean, so an erased patch reads as "average", not black.
        return torch.where(mask, torch.zeros((), device=dev, dtype=x.dtype), x)

    def _augment(self, xb):                          # xb: (B,C,H,W) uint8
        # The full on-device pipeline for one batch: uint8 -> float -> crop -> flip -> normalize
        # -> erase.  Crop/flip/erase only when train=True; normalization always.  Order matters:
        # we crop and flip in raw pixel space, normalize, then erase, so an erased patch lands on
        # the post-normalize mean (i.e. 0).
        x = xb.float()

        # Part A: random crop (a per-sample window out of a reflect-padded image).
        if self.train and self.crop_pad:
            x = self._random_crop(x)

        # Part B: horizontal flip on a random half of the batch.
        #
        # WHY torch.where and not `x[flip] = x[flip].flip(-1)`?
        # -----------------------------------------------------
        # The boolean-mask write runs nonzero() under the hood to turn the mask into indices, and
        # that forces 2 host-device syncs per batch -- a HOST SYNC in the inner loop is exactly
        # what this whole file exists to avoid.  torch.where is a pure element-wise select: same
        # RNG draw, bit-identical output, zero syncs.
        if self.train and self.hflip:
            flip = torch.rand(x.size(0), device=x.device, generator=self.gen) < 0.5   # (B,) bool
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)   # (B,1,1,1) broadcasts over C,H,W

        # Part C: normalize in place -- (x/255 - mean) / std.  mean/std broadcast from (1,3,1,1).
        x = x.div_(255.0).sub_(self.mean).div_(self.std)

        # Part D: random erasing, AFTER normalization (see the ordering note above).
        if self.train and self.erasing:
            x = self._random_erase(x)

        return x

    def __iter__(self):
        n = self.x.size(0)

        # Batch order: a device-side shuffle when training, plain sequential otherwise.  randperm
        # runs on the device with our Generator, so even the shuffle never touches the host.
        idx = (torch.randperm(n, device=self.x.device, generator=self.gen) if self.train
               else torch.arange(n, device=self.x.device))

        # drop_last -> last = n-bs+1, so range() stops before the final short batch (the fixed-shape
        # guarantee the class docstring promised); otherwise last = n and the tail comes along.
        last = n - self.bs + 1 if self.drop_last else n
        for s in range(0, last, self.bs):
            sel = idx[s:s + self.bs]                 # (bs,) indices into the resident tensor
            yield self._augment(self.x[sel]), self.y[sel]
