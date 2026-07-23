from __future__ import annotations

import copy
import importlib
import importlib.util
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch.nn as nn

from .exceptions import InvalidTrialError


@dataclass
class DatasetBundle:
    train: Any
    validation: Any
    test: Any | None
    name: str
    num_classes: int
    train_examples: int
    validation_examples: int
    test_examples: int
    strategy: str
    metadata: dict[str, Any]


class SyntheticResidentDataset:
    """Tiny deterministic dataset with the same epoch API as repository resident datasets."""

    def __init__(
        self,
        *,
        device: torch.device,
        examples: int,
        num_classes: int,
        seed: int,
        train: bool,
        image_size: int = 32,
    ):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.x = torch.randint(
            0,
            256,
            (examples, image_size, image_size, 3),
            dtype=torch.uint8,
            generator=generator,
        ).to(device)
        self.y = torch.randint(
            0,
            num_classes,
            (examples,),
            dtype=torch.long,
            generator=generator,
        ).to(device)
        self.device = device
        self.n = examples
        self.n_classes = num_classes
        self.train = train
        self.mean = torch.tensor((0.5, 0.5, 0.5), device=device).view(1, 3, 1, 1)
        self.std = torch.tensor((0.25, 0.25, 0.25), device=device).view(1, 3, 1, 1)

    def gb(self) -> float:
        return (self.x.numel() + self.y.numel() * self.y.element_size()) / 1e9

    def n_batches(self, batch_size: int) -> int:
        return max(1, self.n // batch_size)

    def epoch(self, batch_size: int, train: bool, generator: torch.Generator | None = None):
        if generator is None:
            generator = torch.Generator(device=self.device).manual_seed(0)
        indices = (
            torch.randperm(self.n, device=self.device, generator=generator)
            if train
            else torch.arange(self.n, device=self.device)
        )
        for start in range(0, self.n, batch_size):
            selected = indices[start : start + batch_size]
            if selected.numel() == 0:
                continue
            x = self.x[selected].permute(0, 3, 1, 2).float().div_(255.0)
            x = x.sub_(self.mean).div_(self.std)
            if train:
                flip = torch.rand(x.size(0), device=x.device, generator=generator) < 0.5
                x = torch.where(flip.view(-1, 1, 1, 1), x.flip(-1), x)
            yield x, self.y[selected]


class ResidentView:
    """Shallow tensor view preserving the repository resident-dataset API."""

    def __init__(self, source: Any, indices: torch.Tensor):
        self._source = source
        self.x = source.x[indices]
        self.y = source.y[indices]
        self.device = source.device
        self.n = int(indices.numel())
        self.n_classes = source.n_classes
        self.mean = source.mean
        self.std = source.std

    def gb(self) -> float:
        return (self.x.numel() * self.x.element_size() + self.y.numel() * self.y.element_size()) / 1e9

    def n_batches(self, batch_size: int) -> int:
        return max(1, self.n // batch_size)

    def epoch(self, batch_size: int, train: bool, generator: torch.Generator | None = None):
        original_x, original_y, original_n = self._source.x, self._source.y, self._source.n
        self._source.x, self._source.y, self._source.n = self.x, self.y, self.n
        try:
            yield from self._source.epoch(batch_size, train, generator=generator)
        finally:
            self._source.x, self._source.y, self._source.n = original_x, original_y, original_n


class RepoModules:
    def __init__(self, repo_root: str | Path):
        self.repo_root = Path(repo_root).resolve()
        self.cv_dir = self.repo_root / "src" / "a1-cv"
        self.common_dir = self.repo_root / "src" / "common"
        if not self.cv_dir.exists():
            raise FileNotFoundError(self.cv_dir)
        for path in (self.cv_dir, self.common_dir):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        self.models = importlib.import_module("models")
        self.train_loop = importlib.import_module("train_loop")
        self.train_run = importlib.import_module("train_run")
        self.cifar_data = importlib.import_module("cifar_data")
        self.imagenet_data = importlib.import_module("imagenet_data")
        self.perfkit = importlib.import_module("perf.perfkit")


def build_trial_model(modules: RepoModules, model_config: dict[str, Any], num_classes: int, device: torch.device) -> nn.Module:
    name = str(model_config.get("name", "resnet18"))
    if name == "vit" and any(
        key in model_config
        for key in ("hidden_dim", "layers", "heads", "mlp_dim", "patch_size")
    ):
        hidden = int(model_config.get("hidden_dim", 384))
        heads = int(model_config.get("heads", 6))
        patch = int(model_config.get("patch_size", 4))
        if hidden % heads:
            raise InvalidTrialError(f"ViT hidden_dim={hidden} must be divisible by heads={heads}")
        if 32 % patch:
            raise InvalidTrialError(f"ViT patch_size={patch} must divide 32")
        model = modules.models.make_vit(
            num_classes=num_classes,
            hidden_dim=hidden,
            layers=int(model_config.get("layers", 6)),
            heads=heads,
            mlp_dim=int(model_config.get("mlp_dim", hidden * 4)),
            patch_size=patch,
        )
    else:
        model = modules.models.build(name, num_classes)
    return model.to(device)


def _stratified_indices(labels: torch.Tensor, validation_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_labels = labels.detach().cpu()
    train_parts: list[torch.Tensor] = []
    validation_parts: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    for label in torch.unique(cpu_labels, sorted=True):
        indices = torch.nonzero(cpu_labels == label, as_tuple=False).flatten()
        permutation = indices[torch.randperm(indices.numel(), generator=generator)]
        count = max(1, int(round(indices.numel() * validation_fraction)))
        validation_parts.append(permutation[:count])
        train_parts.append(permutation[count:])
    train = torch.cat(train_parts).sort().values.to(labels.device)
    validation = torch.cat(validation_parts).sort().values.to(labels.device)
    return train, validation


def subset_view(dataset: Any, fraction: float, *, seed: int, maximum_examples: int | None = None) -> Any:
    target = max(1, int(dataset.n * fraction))
    if maximum_examples is not None:
        target = min(target, maximum_examples)
    if target >= dataset.n:
        return dataset
    generator = torch.Generator(device=dataset.device).manual_seed(seed)
    indices = torch.randperm(dataset.n, device=dataset.device, generator=generator)[:target].sort().values
    return ResidentView(dataset, indices)


def build_trial_dataset(
    modules: RepoModules,
    dataset_config: dict[str, Any],
    runtime_config: dict[str, Any],
    device: torch.device,
) -> DatasetBundle:
    name = str(dataset_config.get("name", "cifar10"))
    seed = int(dataset_config.get("split_seed", runtime_config.get("seed", 42)))
    validation_fraction = float(dataset_config.get("validation_fraction", 0.1))
    strategy = str(dataset_config.get("strategy", "gpu_resident" if device.type != "cpu" else "resident"))

    if name == "synthetic":
        train_examples = int(dataset_config.get("train_examples", 128))
        validation_examples = int(dataset_config.get("validation_examples", 64))
        test_examples = int(dataset_config.get("test_examples", 64))
        classes = int(dataset_config.get("num_classes", 10))
        return DatasetBundle(
            train=SyntheticResidentDataset(device=device, examples=train_examples, num_classes=classes, seed=seed, train=True),
            validation=SyntheticResidentDataset(device=device, examples=validation_examples, num_classes=classes, seed=seed + 1, train=False),
            test=SyntheticResidentDataset(device=device, examples=test_examples, num_classes=classes, seed=seed + 2, train=False),
            name=name,
            num_classes=classes,
            train_examples=train_examples,
            validation_examples=validation_examples,
            test_examples=test_examples,
            strategy="synthetic_resident",
            metadata={"measured": False, "purpose": "software smoke test"},
        )

    if name in {"cifar10", "cifar100"}:
        full = modules.cifar_data.GpuCifar(device, name, "train", seed=seed)
        train_indices, validation_indices = _stratified_indices(full.y, validation_fraction, seed)
        train = ResidentView(full, train_indices)
        validation = ResidentView(full, validation_indices)
        test = modules.cifar_data.GpuCifar(device, name, "test", seed=seed)
        return DatasetBundle(
            train=train,
            validation=validation,
            test=test,
            name=name,
            num_classes=full.n_classes,
            train_examples=train.n,
            validation_examples=validation.n,
            test_examples=test.n,
            strategy=strategy,
            metadata={
                "split_seed": seed,
                "validation_fraction": validation_fraction,
                "test_reserved_for_confirmation": True,
            },
        )

    if name == "imagenet32":
        train = modules.imagenet_data.GpuImageNet32(device, "train", seed=seed)
        validation = modules.imagenet_data.GpuImageNet32(device, "val", seed=seed)
        return DatasetBundle(
            train=train,
            validation=validation,
            test=None,
            name=name,
            num_classes=train.n_classes,
            train_examples=train.n,
            validation_examples=validation.n,
            test_examples=0,
            strategy=strategy,
            metadata={"prepared_path": modules.imagenet_data.DATA_DIR},
        )

    raise ValueError(f"unsupported dataset {name!r}")


def build_trial_optimizer(model: nn.Module, optimizer_config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(optimizer_config.get("optimizer", optimizer_config.get("name", "sgd"))).lower()
    lr = float(optimizer_config.get("learning_rate", optimizer_config.get("lr", 0.1)))
    weight_decay = float(optimizer_config.get("weight_decay", 0.0))
    fused = bool(optimizer_config.get("fused", model.parameters().__iter__().__next__().device.type == "cuda"))
    common = {"lr": lr, "weight_decay": weight_decay}
    if name == "sgd":
        momentum = float(optimizer_config.get("momentum", 0.9))
        dampening = float(optimizer_config.get("dampening", 0.0))
        nesterov = bool(optimizer_config.get("nesterov", momentum > 0 and dampening == 0))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise InvalidTrialError("Nesterov requires momentum > 0 and dampening == 0")
        return torch.optim.SGD(
            model.parameters(),
            momentum=momentum,
            dampening=dampening,
            nesterov=nesterov,
            fused=fused,
            **common,
        )
    if name in {"adam", "adamw"}:
        cls = torch.optim.AdamW if name == "adamw" else torch.optim.Adam
        return cls(
            model.parameters(),
            betas=(float(optimizer_config.get("beta1", 0.9)), float(optimizer_config.get("beta2", 0.999))),
            eps=float(optimizer_config.get("epsilon", optimizer_config.get("eps", 1e-8))),
            amsgrad=bool(optimizer_config.get("amsgrad", False)),
            fused=fused,
            **common,
        )
    raise InvalidTrialError(f"unsupported optimizer {name!r}")


def build_trial_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_config: dict[str, Any],
    total_epochs: int,
):
    name = str(scheduler_config.get("scheduler", scheduler_config.get("name", "cosine"))).lower()
    warmup = int(scheduler_config.get("warmup_epochs", scheduler_config.get("warmup", 0)))
    if name in {"none", "constant"}:
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _epoch: 1.0)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.get("step_size") or max(1, total_epochs // 3)),
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[int(item) for item in scheduler_config.get("milestones") or [total_epochs // 2, int(total_epochs * 0.75)]],
            gamma=float(scheduler_config.get("gamma", 0.1)),
        )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_epochs - warmup),
        eta_min=float(scheduler_config.get("minimum_lr", 0.0)),
    )
    if warmup <= 0:
        return cosine
    linear = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=float(scheduler_config.get("warmup_start_factor", 0.01)),
        total_iters=warmup,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [linear, cosine],
        milestones=[warmup],
    )
