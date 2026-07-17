"""GPU-resident data + augmentation for small images (CIFAR-scale).

Why this exists: a torch DataLoader hands augmentation to CPU worker processes.  That is the
right tool when decode+augment is the bottleneck (see the workers lesson) -- but for 32x32
images a modern GPU trains a small CNN so fast that even 8 CPU workers (~14k img/s) can't keep
it fed, and at batch 128 the model is *launch-bound* anyway.  Measured on the RTX PRO 6000:
CPU-workers bs128 ~13.8k img/s vs GPU-resident bs512 ~18.5k img/s at much higher utilization.

The whole dataset is tiny (CIFAR-100 train = 50k*32*32*3 = 147 MB uint8), so we hold it on the
device as one uint8 NCHW tensor and do crop/flip/erase/normalize on the device in batches.  No
DataLoader, no workers, no host->device copy per batch.  Runs on cuda / mps / cpu unchanged
(it's just tensor ops); it only *pays off* on a GPU, which is exactly when you want it.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F


def to_device_uint8(hf_split, device, img_key='img', label_key='fine_label'):
    """Decode a HuggingFace image split into a device-resident uint8 NCHW tensor + label tensor."""
    imgs = np.stack([np.asarray(im.convert('RGB'), np.uint8) for im in hf_split[img_key]])
    x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).contiguous()  # N,C,H,W uint8
    y = torch.from_numpy(np.asarray(hf_split[label_key], np.int64)).to(device)
    return x, y


class GPUImageLoader:
    """Yield (augmented_float_batch, labels), everything on ``images_u8.device``.

    A drop-in replacement for a DataLoader over small images.  Augmentation (random crop with
    reflect padding, horizontal flip, random erasing) applies only when ``train=True``;
    normalization always.  ``drop_last`` defaults to ``train`` so training batches share a fixed
    shape (keeps cuDNN autotune happy).  Deterministic given ``seed`` (uses a device Generator).
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
        self.drop_last = train if drop_last is None else drop_last
        dev = images_u8.device
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)
        H = images_u8.size(-1)
        self._ar = torch.arange(H, device=dev)
        self.gen = torch.Generator(device=dev)
        if seed is not None:
            self.gen.manual_seed(seed)

    def __len__(self):
        n = self.x.size(0)
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def _random_crop(self, x):                       # x: (B,C,H,W) float, reflect-pad then window
        B, C, H, W = x.shape
        x = F.pad(x, (self.crop_pad,) * 4, mode='reflect')
        m = 2 * self.crop_pad + 1
        oy = torch.randint(0, m, (B,), device=x.device, generator=self.gen)
        ox = torch.randint(0, m, (B,), device=x.device, generator=self.gen)
        rows = (oy.view(B, 1) + self._ar).view(B, 1, H, 1)
        cols = (ox.view(B, 1) + self._ar).view(B, 1, 1, W)
        bidx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
        cidx = torch.arange(C, device=x.device).view(1, C, 1, 1)
        return x[bidx, cidx, rows, cols]             # (B,C,H,W), per-sample crop

    def _random_erase(self, x, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        B, C, H, W = x.shape
        dev = x.device
        do = torch.rand(B, device=dev, generator=self.gen) < self.erase_p
        area = torch.empty(B, device=dev).uniform_(scale[0], scale[1], generator=self.gen) * (H * W)
        logr = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]),
                                                    generator=self.gen)
        ar = torch.exp(logr)
        h = (area * ar).sqrt().round().clamp(1, H).long()
        w = (area / ar).sqrt().round().clamp(1, W).long()
        top = (torch.rand(B, device=dev, generator=self.gen) * (H - h + 1).float()).long()
        left = (torch.rand(B, device=dev, generator=self.gen) * (W - w + 1).float()).long()
        rows = self._ar.view(1, 1, H, 1)
        cols = self._ar.view(1, 1, 1, W)
        mask = ((rows >= top.view(B, 1, 1, 1)) & (rows < (top + h).view(B, 1, 1, 1)) &
                (cols >= left.view(B, 1, 1, 1)) & (cols < (left + w).view(B, 1, 1, 1)) &
                do.view(B, 1, 1, 1))
        return torch.where(mask, torch.zeros((), device=dev, dtype=x.dtype), x)

    def _augment(self, xb):                          # xb: (B,C,H,W) uint8
        x = xb.float()
        if self.train and self.crop_pad:
            x = self._random_crop(x)
        if self.train and self.hflip:
            flip = torch.rand(x.size(0), device=x.device, generator=self.gen) < 0.5
            # torch.where, not x[flip] = x[flip].flip(-1): the boolean-mask write runs nonzero()
            # under the hood, forcing 2 host-device syncs per batch.  Bitwise-identical output.
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
        x = x.div_(255.0).sub_(self.mean).div_(self.std)
        if self.train and self.erasing:
            x = self._random_erase(x)
        return x

    def __iter__(self):
        n = self.x.size(0)
        idx = (torch.randperm(n, device=self.x.device, generator=self.gen) if self.train
               else torch.arange(n, device=self.x.device))
        last = n - self.bs + 1 if self.drop_last else n
        for s in range(0, last, self.bs):
            sel = idx[s:s + self.bs]
            yield self._augment(self.x[sel]), self.y[sel]
