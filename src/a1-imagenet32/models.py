"""
models.py -- the two architectures (ResNet + ViT), each given a 32x32-friendly stem so
torchvision's ImageNet-224 designs actually do useful work on tiny images.

KEY IDEA: these are LITERALLY the builders from cifar10_train.ipynb.  We port them from CIFAR to
ImageNet-32 by moving exactly __one__ thing -- num_classes (10 -> 1000).  Nothing else about the
adaptation changes, because both datasets are 32x32.

WHY does the ResNet stem need surgery at all?
---------------------------------------------
torchvision's ResNet is built for 224x224 ImageNet: a 7x7 stride-2 conv followed by a stride-2
maxpool.  On a 32x32 image that chain would crush the feature map to 8x8 before the network does
any real work, so we swap it for a 3x3 stride-1 conv with __no__ maxpool.  That's just as true on
ImageNet-32 as it was on CIFAR.

WHY patch_size=4 for the ViT?
-----------------------------
Same tiny-image problem.  patch_size=4 gives 32/4 = 8 -> an 8x8 grid = 64 patch tokens (+1 class
token).  The stock patch_size=16 would give a 2x2 grid, i.e. FOUR tokens for a whole image --
nowhere near enough for attention to have anything worth attending to.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import VisionTransformer


def make_resnet18(num_classes: int = 1000) -> nn.Module:
    """torchvision ResNet-18 with its ImageNet-224 stem swapped for a 32x32-friendly one.

    This is the CIFAR baseline (~11.7M params), ported to ImageNet-32 with only num_classes moving.
    """
    # Step 1: build the stock torchvision ResNet-18.
    #
    # WHAT: a standard resnet18 -- no pretrained weights, our own class count.
    # WHY zero_init_residual?  It zeroes the last BatchNorm gamma in each residual block so every
    #   block starts life as an identity map -- the large-batch ResNet trick from Goyal et al.
    #   It's cheap insurance here, and it's exactly what keeps the deeper ResNet-50 below stable.
    m = models.resnet18(weights=None, num_classes=num_classes, zero_init_residual=True)

    # Step 2: 32x32 STEM SURGERY -- swap the whole ImageNet-224 stem for a tiny-image one.
    #
    # WHAT: a 3x3 stride-1 conv (was a 7x7 stride-2), and no maxpool (was a stride-2 maxpool).
    # WHY: the 7x7 stride-2 conv + maxpool would crush a 32x32 image to 8x8 before we've done any
    #   real work.  A 3x3 stride-1 conv with no downsample keeps the full 32x32 resolution.
    # NOTE: bias=False -- the BatchNorm right after conv1 has its own shift, so a conv bias would
    #   just be redundant.
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()          # nn.Identity is a no-op: drops the early downsample
    return m


def make_vit(num_classes: int = 1000, hidden_dim: int = 384, layers: int = 6, heads: int = 6,
             mlp_dim: int = 1536, patch_size: int = 4) -> nn.Module:
    """A ViT sized for 32x32.  Defaults match the CIFAR notebook (~10.7M params on 10 classes).

    Straight torchvision VisionTransformer -- we just feed it 32x32-appropriate hyperparameters.
    """
    # Step 1: hand torchvision's VisionTransformer a configuration sized for 32x32.
    #
    # WHAT: image_size=32 plus the small-model dims (d384, 6 layers, 6 heads).
    # WHY patch_size=4?  This is the ViT half of the stem surgery.  32/4 = 8 -> an 8x8 grid = 64
    #   patch tokens (+1 class token).  The stock patch_size=16 would give a 2x2 grid = only FOUR
    #   tokens for the whole image, far too coarse for attention to do anything useful.
    return VisionTransformer(
        image_size=32,
        patch_size=patch_size,     # 32/4 = 8 -> 8x8 = 64 patch tokens (+1 class token)
        num_layers=layers,
        num_heads=heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        num_classes=num_classes,
    )


def make_resnet50(num_classes: int = 1000) -> nn.Module:
    """A bigger CNN with the same 32x32 stem surgery.  ~23.5M params -- the capacity control for
    the CNN side, so that if the big ViT beats ResNet-18 we know it isn't just 'more parameters
    win'.
    """
    # Step 1: build ResNet-50 -- and here zero_init_residual is load-bearing, not just insurance.
    #
    # WHAT: zero_init_residual initializes the last BN gamma in each block to 0, so every residual
    #   block starts as an identity map -- the standard large-batch ResNet trick (Goyal et al.).
    # IMPORTANT: ResNet-50 actually __needed__ this.  Without it, it hit loss=NaN on epoch 1 at
    #   BOTH lr 0.4 and lr 0.2.  A 50-layer net at batch 512 is simply too unstable to start from
    #   a random residual branch.
    m = models.resnet50(weights=None, num_classes=num_classes, zero_init_residual=True)

    # Step 2: the same 32x32 stem surgery as ResNet-18 -- 3x3 stride-1 conv, no maxpool.
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_vit_base(num_classes: int = 1000) -> nn.Module:
    """A much larger ViT (d768, 12 layers, 12 heads -> ~86M params) -- the "base" config.

    THIS IS THE INTERESTING ONE.  On CIFAR-10 (50k images) a big ViT is hopeless -- there just
    isn't enough data to learn the spatial priors a CNN gets for free.  ImageNet-32 has 1.28M
    images, 25x more.  The whole ViT thesis is that transformers overtake CNNs once the data is
    there, so the question this model asks is: at 1.28M images, has the crossover started?
    """
    # Same builder as make_vit, just scaled up to the "base" width/depth.  patch_size stays 4, so
    # the 8x8 = 64-token grid is unchanged -- only the model gets bigger, not the token count.
    return make_vit(num_classes, hidden_dim=768, layers=12, heads=12, mlp_dim=3072, patch_size=4)


# The model zoo.  Keys are the --model names the training scripts pass; each value is a builder we
# call with num_classes.  The param counts are the trainable totals from n_params() (see below).
BUILDERS = {
    'resnet18': make_resnet18,     # ~11.7M   the CIFAR baseline, ported unchanged
    'resnet50': make_resnet50,     # ~23.5M   bigger CNN, the CNN-side capacity control
    'vit': make_vit,               # ~11.0M   parameter-matched to resnet18
    'vit_base': make_vit_base,     # ~86M     does scale rescue the transformer at 1.28M images?
}


def build(name: str, num_classes: int) -> nn.Module:
    """Look up a builder by name and instantiate it for `num_classes` -- the single entry point
    the training scripts go through."""
    if name not in BUILDERS:
        raise ValueError(f'unknown model {name!r} (expected one of {list(BUILDERS)})')
    return BUILDERS[name](num_classes)


def n_params(m: nn.Module) -> int:
    """Count trainable parameters -- the numbers quoted in BUILDERS come straight from here."""
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
