"""
cifar_data.py: GPU-resident data and augmentation for small (CIFAR-scale) images.

A torch DataLoader hands augmentation off to CPU worker processes. That is the right tool when
decode and augment is the bottleneck (see the workers lesson), but for 32x32 images a modern GPU
trains a small CNN so fast that even 8 CPU workers (about 14k img/s) cannot keep it fed, and at
batch 128 the model is launch-bound anyway. So we flip it around: the dataset is tiny, so we hold
it on the device as one uint8 NCHW tensor and do crop, flip, erase, and normalize right there, in
batches. No DataLoader, no workers, no host-to-device copy per batch.

The payoff, measured on the RTX PRO 6000: CPU-workers bs128 about 13.8k img/s versus GPU-resident
bs512 about 18.5k img/s, at much higher utilization. The CPU pipeline simply cannot feed the card;
the GPU-resident one keeps it saturated.

It is even possible because the dataset is small enough to just live in VRAM: CIFAR-100 train =
50k*32*32*3 = 147 MB as uint8. Keep it uint8 (not float) and it stays cheap, since we convert to
float per batch, which is nearly free on the GPU. This is exactly the trick you cannot pull on
224x224 ImageNet.

It runs on `cuda`, `mps`, or `cpu` unchanged, since it is just tensor ops, but it only pays off on
a GPU, which is precisely when you want it.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

# GpuCifar (defined below) subclasses GpuImageNet32, reusing its resident-batch and augmentation
# machinery unchanged, so importing the base class is all CIFAR needs to add on top of it.
from imagenet_data import GpuImageNet32


def to_device_uint8(hf_split, device, img_key='img', label_key='fine_label'):
    """Decode a HuggingFace image split once, straight onto the device as a uint8 NCHW tensor.

    This is the one-time upload that makes the whole scheme work: we pay the PIL decode and host
    copy exactly once, at startup, and everything after that stays on the device. Labels ride
    along as an int64 tensor on the same device, so nothing in the training loop touches the host.
    """
    imgs = np.stack([np.asarray(im.convert('RGB'), np.uint8) for im in hf_split[img_key]])

    # Reorder the axes from NHWC to NCHW as the images land on the device.
    #
    # imgs: (N, H, W, C)
    # x:    (N, C, H, W), uint8
    #
    # The permute only relabels the strides, so contiguous lays the bytes out as NCHW. The data
    # stays uint8 the whole way.
    x = torch.from_numpy(imgs).to(device).permute(0, 3, 1, 2).contiguous()
    y = torch.from_numpy(np.asarray(hf_split[label_key], np.int64)).to(device)
    return x, y


class GPUImageLoader:
    """A DataLoader-shaped iterable over small images that never leaves the device.

    Yields `(augmented_float_batch, labels)`, everything on `images_u8.device`. The augmentation
    (random crop with reflect padding, horizontal flip, random erasing) fires only when
    `train=True`; normalization always runs. `drop_last` defaults to `train` so every training
    batch shares one fixed shape, which keeps cuDNN's autotuner from re-planning on a short tail
    batch. Deterministic given `seed`: it draws from a device Generator, so even the RNG lives on
    the device.
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

        # drop_last defaults to train so every training batch shares one fixed shape.
        self.drop_last = train if drop_last is None else drop_last
        dev = images_u8.device

        # Shape mean and std as (1, 3, 1, 1) so they broadcast cleanly over an (N, 3, H, W) batch.
        self.mean = torch.tensor(mean, device=dev).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=dev).view(1, 3, 1, 1)

        # A shared 0..H-1 ramp, reused to build the crop and erase index masks (H == W here).
        H = images_u8.size(-1)
        self._ar = torch.arange(H, device=dev)

        # A device Generator: the same seed gives the same augmentation, and every draw stays on the
        # device, so sampling never forces a host sync.
        self.gen = torch.Generator(device=dev)
        if seed is not None:
            self.gen.manual_seed(seed)

    def __len__(self):
        n = self.x.size(0)
        # With drop_last, floor division keeps whole batches only; otherwise ceil-div keeps the tail.
        return n // self.bs if self.drop_last else (n + self.bs - 1) // self.bs

    def _random_crop(self, x):
        # RandomCrop(H, padding=crop_pad) for a whole batch at once. Reflect-pad the image, then
        # pull a per-sample HxW window back out with advanced indexing, so every image in the batch
        # gets its own random offset in a single gather, with no Python loop.
        #
        # x: (B, C, H, W) float
        #
        # B is the batch, C the channels, H and W the height and width, and m the number of valid
        # crop offsets.
        B, C, H, W = x.shape

        # Step 1: Reflect-pad all four sides.
        #
        # x (before): (B, C, H, W)
        # x (after):  (B, C, H+2p, W+2p)
        #
        # Reflect (not zero) keeps the border as real image content instead of a black frame the
        # model could learn to key off.
        x = F.pad(x, (self.crop_pad,) * 4, mode='reflect')


        # Step 2: Draw a random top-left corner per sample.
        #
        # oy: (B,) row offsets
        # ox: (B,) col offsets
        #
        # After padding there are m = 2p+1 valid offsets along each axis (0 to 2p inclusive).
        m = 2 * self.crop_pad + 1
        oy = torch.randint(0, m, (B,), device=x.device, generator=self.gen)
        ox = torch.randint(0, m, (B,), device=x.device, generator=self.gen)


        # Step 3: Turn each corner into the H (or W) absolute indices it selects.
        #
        # rows: (B, 1, H, 1)
        # cols: (B, 1, 1, W)
        #
        # Broadcasting the shared ramp against the per-sample offset, oy.view(B, 1) + arange(H)
        # gives (B, H), then reshaped so rows and cols land on separate axes.
        rows = (oy.view(B, 1) + self._ar).view(B, 1, H, 1)
        cols = (ox.view(B, 1) + self._ar).view(B, 1, 1, W)


        # Step 4: Build batch and channel index tensors, shaped to broadcast against rows and cols.
        #
        # bidx: (B, 1, 1, 1)
        # cidx: (1, C, 1, 1)
        bidx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
        cidx = torch.arange(C, device=x.device).view(1, C, 1, 1)


        # Step 5: Gather the crop.
        #
        # result: (B, C, H, W), per-sample crop
        #
        # All four index tensors broadcast to (B, C, H, W), so the whole batch is cropped in one
        # kernel. The equivalent loop would be:
        #   for b in range(B): out[b] = x[b, :, oy[b]:oy[b]+H, ox[b]:ox[b]+W]
        return x[bidx, cidx, rows, cols]

    def _random_erase(self, x, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        # RandomErasing for the whole batch with no Python loop. We sample a rectangle per image,
        # then paint the erase region as a boolean mask via broadcast comparisons. Looping in
        # Python over just the ~erase_p fraction that gets erased is the trap: a few hundred tiny
        # GPU ops per batch is pure launch overhead (the same launch-bound lesson from the module
        # header). One masked write over the batch is a handful of kernels regardless of B.
        #
        # B is the batch, C the channels, H and W the height and width; h and w the erase-rectangle
        # size per sample, (top, left) its corner, and `do` which samples get erased at all.
        B, C, H, W = x.shape
        dev = x.device

        # Step 1: Decide who gets erased, Bernoulli(erase_p) per sample.
        #
        # do: (B,) bool
        #
        # Samples with do=False drop out of the mask below and are left untouched.
        do = torch.rand(B, device=dev, generator=self.gen) < self.erase_p

        # Step 2: Pick a target area (a fraction of H*W) and an aspect ratio per sample.
        #
        # area: (B,)
        # logr: (B,)
        # ar:   (B,) aspect ratio, back in linear space
        #
        # Ratio is sampled in log space so that e.g. 3:1 and 1:3 are equally likely; a uniform draw
        # on the raw ratio would bias toward wide rectangles.
        area = torch.empty(B, device=dev).uniform_(scale[0], scale[1], generator=self.gen) * (H * W)
        logr = torch.empty(B, device=dev).uniform_(math.log(ratio[0]), math.log(ratio[1]),
                                                    generator=self.gen)
        ar = torch.exp(logr)

        # Step 3: Convert area and ratio into an integer rectangle size, clamped to fit inside the
        # image.
        #
        # h: (B,) rect height
        # w: (B,) rect width
        h = (area * ar).sqrt().round().clamp(1, H).long()
        w = (area / ar).sqrt().round().clamp(1, W).long()

        # Step 4: Pick a random top-left corner that keeps the rectangle in bounds.
        #
        # top:  (B,)
        # left: (B,)
        #
        # (H - h + 1) is the count of valid top rows; scale a uniform [0, 1) by it and floor.
        top = (torch.rand(B, device=dev, generator=self.gen) * (H - h + 1).float()).long()
        left = (torch.rand(B, device=dev, generator=self.gen) * (W - w + 1).float()).long()

        # Step 5: Build the erase mask.
        #
        # rows: (1, 1, H, 1)
        # cols: (1, 1, 1, W)
        # mask: (B, 1, H, W), broadcasts over C
        #
        # Compare the shared row and col ramp against each sample's [top, top+h) x [left, left+w)
        # box, then apply `do` with a logical and to keep only the chosen samples. The corners are
        # shaped (B, 1, 1, 1).
        rows = self._ar.view(1, 1, H, 1)
        cols = self._ar.view(1, 1, 1, W)
        mask = ((rows >= top.view(B, 1, 1, 1)) & (rows < (top + h).view(B, 1, 1, 1)) &
                (cols >= left.view(B, 1, 1, 1)) & (cols < (left + w).view(B, 1, 1, 1)) &
                do.view(B, 1, 1, 1))

        # Step 6: Select 0 where the mask is True and x elsewhere, element-wise. torch.where (not an
        # in-place masked write) keeps this a pure element-wise kernel with no host sync. 0 is the
        # post-normalize channel mean, so an erased patch reads as "average", not black.
        return torch.where(mask, torch.zeros((), device=dev, dtype=x.dtype), x)

    def _augment(self, xb):
        # The full on-device pipeline for one batch: uint8 to float, then crop, flip, normalize,
        # and erase. Crop, flip, and erase run only when train=True; normalization always runs.
        # Order matters: we crop and flip in raw pixel space, normalize, then erase, so an erased
        # patch lands on the post-normalize mean (that is, 0).
        #
        # xb: (B, C, H, W) uint8
        x = xb.float()

        # Part A: Random crop, a per-sample window out of a reflect-padded image.
        if self.train and self.crop_pad:
            x = self._random_crop(x)

        # Part B: Horizontal flip on a random half of the batch.
        #
        # flip: (B,) bool
        #
        # We use torch.where rather than `x[flip] = x[flip].flip(-1)` because the boolean-mask write
        # runs nonzero() under the hood to turn the mask into indices, and that forces 2 host-device
        # syncs per batch. A host sync in the inner loop is exactly what this whole file exists to
        # avoid. torch.where is a pure element-wise select: same RNG draw, bit-identical output,
        # zero syncs. The (B, 1, 1, 1) mask broadcasts over C, H, and W.
        if self.train and self.hflip:
            flip = torch.rand(x.size(0), device=x.device, generator=self.gen) < 0.5
            x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)

        # Part C: Normalize in place, (x/255 - mean) / std. The mean and std broadcast from
        # (1, 3, 1, 1).
        x = x.div_(255.0).sub_(self.mean).div_(self.std)

        # Part D: Random erasing, after normalization (see the ordering note above).
        if self.train and self.erasing:
            x = self._random_erase(x)

        return x

    def __iter__(self):
        n = self.x.size(0)

        # Batch order: a device-side shuffle when training, plain sequential otherwise. randperm
        # runs on the device with our Generator, so even the shuffle never touches the host.
        idx = (torch.randperm(n, device=self.x.device, generator=self.gen) if self.train
               else torch.arange(n, device=self.x.device))

        # With drop_last, last = n-bs+1, so range() stops before the final short batch (the
        # fixed-shape guarantee the class docstring promised); otherwise last = n and the tail comes
        # along.
        last = n - self.bs + 1 if self.drop_last else n
        for s in range(0, last, self.bs):
            # sel: (bs,) indices into the resident tensor
            sel = idx[s:s + self.bs]
            yield self._augment(self.x[sel]), self.y[sel]


# Published per-channel train-split statistics, plus the class count, for each dataset. We use the
# standard constants rather than recomputing them so a run normalizes CIFAR exactly as the notebooks
# and the literature do.
_CIFAR_STATS = {
    'cifar10':  ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10),
    'cifar100': ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100),
}


class GpuCifar(GpuImageNet32):
    """CIFAR-10 or CIFAR-100, held resident on the GPU, serving the same augmented batches as
    GpuImageNet32.

    CIFAR is 32x32 just like ImageNet-32, so there is nothing new to write for the inner loop: this
    inherits GpuImageNet32's epoch(), n_batches(), _to_float(), and batched crop/flip augmentation
    unchanged, and the module-level mixup_cutmix / random_erasing_ work on a CIFAR batch as-is. All
    that differs is where the pixels come from, so __init__ loads torchvision's CIFAR (whose .data is
    already (N, 32, 32, 3) uint8, the exact layout GpuImageNet32 expects) and sets the same handful of
    attributes the inherited methods read. It deliberately does not call super().__init__, which would
    look for the ImageNet-32 arrays.
    """

    def __init__(self, device: torch.device, dataset: str = 'cifar10', split: str = 'train',
                 subset: int | None = None, seed: int = 0):
        from datasets import load_dataset

        if dataset not in _CIFAR_STATS:
            raise ValueError(f'unknown dataset {dataset!r} (expected cifar10 or cifar100)')
        mean, std, n_classes = _CIFAR_STATS[dataset]

        # We load from HuggingFace, whose CDN is fast and cached, so this is a one-time download and
        # instant after. We deliberately avoid torchvision's CIFAR here: its download mirror crawls,
        # and a truncated local CIFAR-100 tarball sends it into a half-hour re-download. CIFAR-100
        # keys its labels as 'fine_label' (there is also a 20-way 'coarse_label' we ignore).
        hf_name = 'uoft-cs/cifar10' if dataset == 'cifar10' else 'uoft-cs/cifar100'
        label_key = 'label' if dataset == 'cifar10' else 'fine_label'
        sp = load_dataset(hf_name, split=('train' if split == 'train' else 'test'))

        # Decode the PIL images once into one (N, 32, 32, 3) uint8 array: NHWC, the exact layout
        # GpuImageNet32 stores and its _to_float() expects.
        x = np.stack([np.asarray(im.convert('RGB'), np.uint8) for im in sp['img']])
        y = np.asarray(sp[label_key], np.int64)

        # A random subset for smoke tests. CIFAR files are not class-sorted the way ImageNet-32's are,
        # but we sample at random anyway so a subset stays representative of every class.
        if subset is not None and subset < len(x):
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(len(x), size=subset, replace=False))
            x, y = x[idx], y[idx]

        # Set exactly the attributes the inherited methods read: pixels resident as uint8, labels as
        # int64, the normalization constants shaped to broadcast, and the class count.
        self.x = torch.from_numpy(np.ascontiguousarray(x)).to(device)
        self.y = torch.from_numpy(y).to(device)
        self.device = device
        self.n = len(self.y)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)
        self.n_classes = n_classes
