from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .baselines import Baseline


def normalized_parameter_distance(
    discovered: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Compare common hyperparameters without claiming universal optimality."""
    ranges = ranges or {}
    components: dict[str, float | None] = {}
    for name in sorted(set(discovered) & set(reference)):
        left, right = discovered[name], reference[name]
        if isinstance(left, bool) or isinstance(right, bool):
            components[name] = 0.0 if left == right else 1.0
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            low, high = ranges.get(name, (min(float(left), float(right)), max(float(left), float(right))))
            scale = max(abs(high - low), 1e-12)
            components[name] = abs(float(left) - float(right)) / scale
        elif isinstance(left, str) and isinstance(right, str):
            components[name] = 0.0 if left == right else 1.0
        else:
            components[name] = None
    numeric = [value for value in components.values() if value is not None]
    return {
        "components": components,
        "mean_normalized_distance": sum(numeric) / len(numeric) if numeric else None,
        "compared_parameters": len(numeric),
    }


def compare_with_reference(
    discovered_row: Mapping[str, Any],
    reference: Baseline,
    *,
    accuracy_margin: float = 0.01,
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    metrics = discovered_row.get("metrics", discovered_row)
    params = discovered_row.get("params", {})
    discovered_accuracy = metrics.get("validation_top1")
    accuracy_gap = None
    within_margin = None
    if discovered_accuracy is not None and reference.validation_top1 is not None:
        accuracy_gap = float(discovered_accuracy) - float(reference.validation_top1)
        within_margin = accuracy_gap >= -abs(float(accuracy_margin))
    return {
        "reference_name": reference.name,
        "reference_source": reference.source,
        "reference_measured": reference.measured,
        "reference_validation_top1": reference.validation_top1,
        "discovered_validation_top1": discovered_accuracy,
        "accuracy_gap": accuracy_gap,
        "within_configured_margin": within_margin,
        "parameter_distance": normalized_parameter_distance(params, reference.params, ranges=ranges),
        "caveats": [
            "the repository result used the final CIFAR test split during training-time evaluation",
            "the HPO adapter uses a deterministic train/validation split and reserves test data",
            "hardware, fidelity, seeds, augmentation, and epoch budgets may differ",
        ],
    }


def _ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def proxy_reliability(proxy_scores: Mapping[str, float], full_scores: Mapping[str, float], *, top_k: int = 3) -> dict[str, Any]:
    common = sorted(set(proxy_scores) & set(full_scores))
    if len(common) < 2:
        return {"sample_count": len(common), "rank_correlation": None, "top_k_retention": None}
    proxy = [float(proxy_scores[key]) for key in common]
    full = [float(full_scores[key]) for key in common]
    proxy_ranks = _ranks(proxy)
    full_ranks = _ranks(full)
    mean_p = sum(proxy_ranks) / len(proxy_ranks)
    mean_f = sum(full_ranks) / len(full_ranks)
    numerator = sum((p - mean_p) * (f - mean_f) for p, f in zip(proxy_ranks, full_ranks, strict=True))
    denominator = math.sqrt(
        sum((p - mean_p) ** 2 for p in proxy_ranks)
        * sum((f - mean_f) ** 2 for f in full_ranks)
    )
    correlation = numerator / denominator if denominator else None
    limit = min(max(1, top_k), len(common))
    proxy_top = set(sorted(common, key=lambda key: proxy_scores[key], reverse=True)[:limit])
    full_top = set(sorted(common, key=lambda key: full_scores[key], reverse=True)[:limit])
    return {
        "sample_count": len(common),
        "rank_correlation": correlation,
        "top_k": limit,
        "top_k_retention": len(proxy_top & full_top) / limit,
        "false_elimination_rate": len(full_top - proxy_top) / limit,
    }


def discovery_benchmark_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference: Baseline | None = None,
    thresholds: Sequence[float] = (0.8, 0.85, 0.9),
    sampled_trials: int | None = None,
    pruned_trials: int | None = None,
    promoted_trials: int | None = None,
    parameter_ranges: Mapping[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    completed = [row for row in rows if row.get("status") == "completed" and row.get("metrics")]
    ordered = sorted(completed, key=lambda row: float(row.get("finished_at", math.inf)))
    best = max(ordered, key=lambda row: float(row["metrics"].get("validation_top1", -math.inf)), default=None)
    start = min((float(row.get("started_at", 0.0)) for row in ordered), default=0.0)
    time_to_threshold: dict[str, float | None] = {}
    for threshold in thresholds:
        reached = next(
            (
                row for row in ordered
                if float(row["metrics"].get("validation_top1", -math.inf)) >= float(threshold)
            ),
            None,
        )
        time_to_threshold[str(threshold)] = (
            None if reached is None else max(0.0, float(reached.get("finished_at", start)) - start)
        )
    comparison = None
    reference_dominated = None
    if reference is not None and best is not None:
        comparison = compare_with_reference(
            best,
            reference,
            ranges=parameter_ranges,
        )
        if reference.validation_top1 is not None and reference.seconds is not None:
            reference_dominated = any(
                float(row["metrics"].get("validation_top1", -math.inf)) >= reference.validation_top1
                and float(row["metrics"].get("wall_seconds", math.inf)) <= reference.seconds
                and (
                    float(row["metrics"].get("validation_top1", -math.inf)) > reference.validation_top1
                    or float(row["metrics"].get("wall_seconds", math.inf)) < reference.seconds
                )
                for row in ordered
            )
    return {
        "sampled_trials": sampled_trials if sampled_trials is not None else len(rows),
        "completed_trials": len(completed),
        "pruned_trials": pruned_trials,
        "promoted_trials": promoted_trials,
        "best_validation_top1": None if best is None else best["metrics"].get("validation_top1"),
        "best_candidate_id": None if best is None else best.get("candidate_id"),
        "time_to_accuracy_threshold_seconds": time_to_threshold,
        "total_examples": sum(int(row["metrics"].get("total_examples", 0) or 0) for row in completed),
        "total_epochs": sum(float(row["metrics"].get("epochs_completed", 0) or 0) for row in completed),
        "gpu_hours": sum(float(row["metrics"].get("gpu_hours", 0) or 0) for row in completed),
        "cpu_hours": sum(float(row["metrics"].get("cpu_hours", 0) or 0) for row in completed),
        "peak_memory_mb": max((float(row["metrics"].get("peak_gpu_memory_mb", 0) or 0) for row in completed), default=0.0),
        "reference_comparison": comparison,
        "reference_pareto_dominated_on_accuracy_time": reference_dominated,
    }
