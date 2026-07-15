"""
models.py -- the two architectures, adapted for 32x32 inputs.

These are LITERALLY the functions from cifar10_train.ipynb, with `num_classes` as an argument.
Nothing about the adaptation changes between CIFAR-10 and ImageNet-32, because both are 32x32:

  * ResNet's stem surgery (7x7 stride-2 conv + maxpool -> 3x3 stride-1 conv, no maxpool) is needed
    because torchvision's ResNet targets 224x224 ImageNet and would crush a 32x32 image to 8x8
    before doing any real work.  That is just as true here.
  * The ViT's patch_size=4 (-> an 8x8 grid = 64 tokens) is needed for the same reason: the stock
    patch_size=16 would give a 2x2 grid, i.e. FOUR tokens for a whole image.

The only thing that moves between the two datasets is num_classes: 10 -> 1000.
"""
from __future__ import annotations

import torch.nn as nn
from torchvision import models
from torchvision.models import VisionTransformer


def make_resnet18(num_classes: int = 1000) -> nn.Module:
    """torchvision ResNet-18 with an ImageNet-224 stem swapped for a 32x32-friendly one."""
    m = models.resnet18(weights=None, num_classes=num_classes, zero_init_residual=True)
    # 3x3 stride-1, no bias (the BatchNorm right after it has its own shift, so a bias is redundant).
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()          # drop the early downsample
    return m


def make_vit(num_classes: int = 1000, hidden_dim: int = 384, layers: int = 6, heads: int = 6,
             mlp_dim: int = 1536, patch_size: int = 4) -> nn.Module:
    """A ViT sized for 32x32.  Defaults match the CIFAR notebook (~10.7M params on 10 classes)."""
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
    """A bigger CNN, same 32x32 stem surgery.  ~23.5M params -- the capacity control for the CNN
    side, so that if the big ViT beats ResNet-18 we know it is not just 'more parameters win'."""
    # zero_init_residual: initialize the last BN gamma in each block to 0, so every residual block
    # starts as an identity map.  This is the standard large-batch ResNet trick (Goyal et al.), and
    # it is what ResNet-50 needed: without it, it hit loss=NaN on epoch 1 at BOTH lr 0.4 and lr 0.2.
    # A 50-layer net at batch 512 is simply too unstable to start from a random residual branch.
    m = models.resnet50(weights=None, num_classes=num_classes, zero_init_residual=True)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    return m


def make_vit_base(num_classes: int = 1000) -> nn.Module:
    """A much larger ViT (d768, 12 layers, 12 heads -> ~86M params).

    THIS IS THE INTERESTING ONE.  On CIFAR-10 (50k images) a big ViT would be hopeless -- not enough
    data to learn the spatial priors a CNN gets for free.  ImageNet-32 has 1.28M images, 25x more.
    The whole ViT thesis is that transformers overtake CNNs once the data is there, so the question
    this model asks is: at 1.28M images, has the crossover started?
    """
    return make_vit(num_classes, hidden_dim=768, layers=12, heads=12, mlp_dim=3072, patch_size=4)


BUILDERS = {
    'resnet18': make_resnet18,     # ~11.7M   the CIFAR baseline, ported unchanged
    'resnet50': make_resnet50,     # ~23.5M   bigger CNN
    'vit': make_vit,               # ~11.0M   parameter-matched to resnet18
    'vit_base': make_vit_base,     # ~86M     does scale rescue the transformer at 1.28M images?
}


def build(name: str, num_classes: int) -> nn.Module:
    if name not in BUILDERS:
        raise ValueError(f'unknown model {name!r} (expected one of {list(BUILDERS)})')
    return BUILDERS[name](num_classes)


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
