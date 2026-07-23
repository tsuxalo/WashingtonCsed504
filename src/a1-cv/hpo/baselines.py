from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Baseline:
    name: str
    dataset: str
    model: str
    params: dict[str, Any]
    validation_top1: float | None
    seconds: float | None
    source: str
    measured: bool


REPOSITORY_RECIPES = {
    ("cifar10", "resnet18"): {
        "optimizer": "sgd",
        "batch_size": 512,
        "learning_rate": 0.2,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": True,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "gradient_clip": 0.0,
        "augmentation": "basic",
        "epochs": 30,
    },
    ("cifar100", "resnet18"): {
        "optimizer": "sgd",
        "batch_size": 512,
        "learning_rate": 0.2,
        "weight_decay": 5e-4,
        "momentum": 0.9,
        "nesterov": True,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "label_smoothing": 0.1,
        "gradient_clip": 0.0,
        "augmentation": "basic",
        "epochs": 40,
    },
    ("cifar10", "vit"): {
        "optimizer": "adamw",
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.05,
        "scheduler": "cosine",
        "warmup_epochs": 5,
        "gradient_clip": 1.0,
        "augmentation": "strong",
        "epochs": 200,
    },
}


def load_repository_baselines(repo_root: str | Path) -> list[Baseline]:
    root = Path(repo_root) / "src" / "a1-cv" / "runs"
    results = []
    for path in sorted(root.glob("*_result.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        dataset = row.get("dataset") or ("imagenet32" if path.name.startswith("imagenet32") else "unknown")
        recipe = REPOSITORY_RECIPES.get((dataset, row.get("model")), {})
        results.append(Baseline(
            name=str(row.get("tag", path.stem)),
            dataset=dataset,
            model=str(row.get("model")),
            params={**recipe, "batch_size": row.get("batch"), "learning_rate": row.get("lr"), "epochs": row.get("epochs")},
            validation_top1=row.get("best_top1"),
            seconds=row.get("seconds"),
            source=str(path.relative_to(repo_root)),
            measured=True,
        ))
    return results


def enqueue_reference(study: Any, baseline: Baseline) -> None:
    study.enqueue_trial(dict(baseline.params), user_attrs={"reference": baseline.name, "source": baseline.source})
