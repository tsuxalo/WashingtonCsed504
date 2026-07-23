from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from .objectives import metric_value
from .schemas import ObjectiveSpec


def dominates(left: Mapping[str, Any], right: Mapping[str, Any], objectives: Sequence[ObjectiveSpec]) -> bool:
    at_least_as_good = True
    strictly_better = False
    for objective in objectives:
        a = metric_value(left, objective.name)
        b = metric_value(right, objective.name)
        if objective.direction == "maximize":
            at_least_as_good &= a >= b
            strictly_better |= a > b
        else:
            at_least_as_good &= a <= b
            strictly_better |= a < b
    return at_least_as_good and strictly_better


def pareto_front(rows: Iterable[Mapping[str, Any]], objectives: Sequence[ObjectiveSpec]) -> list[dict[str, Any]]:
    values = [dict(row) for row in rows]
    return [
        row
        for index, row in enumerate(values)
        if not any(
            dominates(other, row, objectives)
            for other_index, other in enumerate(values)
            if other_index != index
        )
    ]


def highest_accuracy_under_budget(rows, *, budget_name: str, maximum: float):
    feasible = [row for row in rows if float(row["metrics"].get(budget_name, math.inf)) <= maximum]
    return max(feasible, key=lambda row: float(row["metrics"].get("validation_top1", -math.inf)), default=None)


def fastest_above_accuracy(rows, *, minimum_accuracy: float):
    feasible = [row for row in rows if float(row["metrics"].get("validation_top1", 0)) >= minimum_accuracy]
    return min(feasible, key=lambda row: float(row["metrics"].get("wall_seconds", math.inf)), default=None)


def lowest_memory_above_accuracy(rows, *, minimum_accuracy: float):
    feasible = [row for row in rows if float(row["metrics"].get("validation_top1", 0)) >= minimum_accuracy]
    return min(feasible, key=lambda row: float(row["metrics"].get("peak_gpu_memory_mb", math.inf)), default=None)


def pareto_knee(rows, objectives: Sequence[ObjectiveSpec]):
    values = [dict(row) for row in rows]
    metric_rows = [dict(row.get("metrics", {}), _row=row) for row in values]
    front_metrics = pareto_front(metric_rows, objectives)
    front = [item["_row"] for item in front_metrics]
    if not front:
        return None
    normalized: dict[str, tuple[float, float]] = {}
    for objective in objectives:
        objective_values = [metric_value(row["metrics"], objective.name) for row in front]
        normalized[objective.name] = (min(objective_values), max(objective_values))

    def score(row):
        total = 0.0
        for objective in objectives:
            low, high = normalized[objective.name]
            value = metric_value(row["metrics"], objective.name)
            unit = 0.5 if high == low else (value - low) / (high - low)
            if objective.direction == "minimize":
                unit = 1 - unit
            total += (1 - unit) ** 2
        return total ** 0.5

    return min(front, key=score)


def lowest_cost_within_accuracy_margin(rows, *, accuracy_margin: float = 0.01):
    values = list(rows)
    if not values:
        return None
    best_accuracy = max(float(row["metrics"].get("validation_top1", -math.inf)) for row in values)
    threshold = best_accuracy - float(accuracy_margin)
    feasible = [
        row for row in values
        if float(row["metrics"].get("validation_top1", -math.inf)) >= threshold
        and row["metrics"].get("estimated_cost_usd") is not None
    ]
    return min(
        feasible,
        key=lambda row: float(row["metrics"].get("estimated_cost_usd", math.inf)),
        default=None,
    )
