from __future__ import annotations

from typing import Any, Mapping, Sequence

from .schemas import ObjectiveSpec


_METRIC_ALIASES = {
    "validation_accuracy": "validation_top1",
    "validation_top1_accuracy": "validation_top1",
    "trial_wall_time": "wall_seconds",
    "trial_duration": "wall_seconds",
    "peak_memory": "peak_gpu_memory_mb",
    "cost": "estimated_cost_usd",
    "examples_processed": "total_examples",
    "flops": "approximate_training_flops",
}


def metric_value(metrics: Mapping[str, Any], name: str) -> float:
    key = _METRIC_ALIASES.get(name, name)
    value = metrics.get(key)
    if value is None:
        raise KeyError(f"objective metric {name!r} is unavailable")
    return float(value)


def objective_values(metrics: Mapping[str, Any], objectives: Sequence[ObjectiveSpec]) -> list[float]:
    return [metric_value(metrics, objective.name) for objective in objectives]


def primary_score(metrics: Mapping[str, Any], objective: ObjectiveSpec) -> float:
    value = metric_value(metrics, objective.name)
    return value if objective.direction == "maximize" else -value
