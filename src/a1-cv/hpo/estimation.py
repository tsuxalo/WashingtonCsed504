from __future__ import annotations

import copy
import math
import statistics
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

from .schemas import StudyConfig


@dataclass
class EstimateRange:
    optimistic: float
    expected: float
    conservative: float
    unit: str
    method: str


@dataclass
class WorkEstimate:
    mode: str
    expected_trials: int
    expected_pruned_trials: int
    expected_promoted_trials: int
    expected_full_evaluations: int
    expected_epochs: float
    expected_steps: float
    expected_examples: float
    duration_seconds: EstimateRange
    gpu_hours: EstimateRange
    cpu_hours: EstimateRange
    flops: EstimateRange | None
    peak_memory_mb: float | None
    storage_mb: float | None
    monetary_cost_usd: EstimateRange | None
    assumptions: list[str]
    classification: str = "calibrated projection"

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result


def _range(value: float, low: float = 0.75, high: float = 1.6, *, unit: str, method: str) -> EstimateRange:
    return EstimateRange(value * low, value, value * high, unit, method)


def _steps(examples: float, batch_size: int) -> float:
    return math.ceil(examples / max(1, batch_size))


def estimate_search(
    config: StudyConfig,
    *,
    train_examples: int,
    validation_examples: int,
    calibration_records: Sequence[Mapping[str, Any]],
    representative_batch_size: int | None = None,
) -> WorkEstimate:
    completed = [row for row in calibration_records if row.get("seconds_per_example")]
    if not completed:
        raise ValueError("at least one calibration record with seconds_per_example is required")
    seconds_per_example = statistics.median(float(row["seconds_per_example"]) for row in completed)
    eval_seconds = statistics.median(float(row.get("evaluation_seconds_per_epoch", 0.0)) for row in completed)
    checkpoint_seconds = statistics.median(float(row.get("checkpoint_seconds_per_epoch", 0.0)) for row in completed)
    checkpoint_mb = statistics.median(float(row.get("checkpoint_size_mb", 0.0)) for row in completed)
    peak_memory = max(float(row.get("peak_memory_mb", 0.0)) for row in completed)
    train_flops_per_image = next((float(row["train_flops_per_image"]) for row in completed if row.get("train_flops_per_image")), None)
    batch_size = representative_batch_size or int(statistics.median(float(row.get("batch_size", 128)) for row in completed))
    assumptions = [
        "duration is projected from median measured seconds per training example",
        "validation and checkpoint overhead use calibration medians",
        "successive-halving continuation counts only incremental resource",
        "actual sampled architecture, optimizer, and batch size can shift runtime",
    ]

    trials = config.full.trials if config.mode == "full" else config.proxy.trials
    proxy_examples = proxy_epochs = 0.0
    halving_examples = halving_epochs = 0.0
    full_examples = full_epochs = 0.0
    promoted = pruned = full_evaluations = 0

    if config.mode in {"proxy", "successive_halving"}:
        budget = config.proxy.budget
        proxy_epoch_limit = budget.epochs or 1
        per_trial_examples = train_examples * budget.data_fraction * proxy_epoch_limit
        if budget.max_steps is not None:
            per_trial_examples = min(per_trial_examples, budget.max_steps * batch_size)
        proxy_examples = trials * per_trial_examples
        proxy_epochs = trials * min(proxy_epoch_limit, per_trial_examples / (train_examples * budget.data_fraction))

    if config.mode == "proxy":
        promoted = min(config.proxy.promote_top_k, trials)
        pruned = max(0, trials - promoted)
    elif config.mode == "successive_halving":
        active = min(config.proxy.promote_top_k, trials)
        previous = config.proxy.budget.epochs or 0
        for budget in config.successive_halving.rung_budgets:
            incremental = max(0.0, float(budget) - float(previous))
            halving_examples += active * incremental * train_examples
            halving_epochs += active * incremental
            survivors = max(
                config.successive_halving.minimum_trials_per_rung,
                math.ceil(active / config.successive_halving.reduction_factor),
            )
            pruned += max(0, active - survivors)
            promoted += survivors
            active = survivors
            previous = float(budget)
        if config.full.enabled:
            full_evaluations = min(active, config.full.trials)
            seed_count = len(config.full.budget.seeds)
            full_epochs = full_evaluations * (config.full.budget.epochs or 1) * seed_count
            full_examples = full_epochs * train_examples * config.full.budget.data_fraction
    else:
        full_evaluations = trials
        seed_count = len(config.full.budget.seeds)
        full_epochs = trials * (config.full.budget.epochs or 1) * seed_count
        full_examples = full_epochs * train_examples * config.full.budget.data_fraction

    total_examples = proxy_examples + halving_examples + full_examples
    total_epochs = proxy_epochs + halving_epochs + full_epochs
    total_steps = _steps(total_examples, batch_size)
    overhead = total_epochs * (eval_seconds + checkpoint_seconds)
    expected_seconds = total_examples * seconds_per_example + overhead
    if config.mode == "full":
        pruned = 0
        promoted = 0
    gpu_fraction = 1.0 if config.runtime.device in {"auto", "cuda"} else 0.0
    cpu_ratio = statistics.median(
        float(row.get("cpu_seconds", 0.0)) / max(float(row.get("elapsed_seconds", 1.0)), 1e-9)
        for row in completed
    ) if any(row.get("cpu_seconds") for row in completed) else 0.1
    flops_value = None if train_flops_per_image is None else total_examples * train_flops_per_image
    storage = checkpoint_mb * max(1, trials + promoted + full_evaluations)
    duration = _range(expected_seconds, unit="seconds", method="calibrated projection")
    gpu_hours = _range(expected_seconds * gpu_fraction / 3600, unit="GPU-hours", method="calibrated projection")
    cpu_hours = _range(expected_seconds * cpu_ratio / 3600, unit="CPU-hours", method="heuristic from calibration")
    flops = None if flops_value is None else _range(flops_value, 0.9, 1.2, unit="FLOPs", method="repository FlopCounterMode projection")

    rates = config.cost_rates
    expected_cost = 0.0
    missing = False
    if gpu_hours.expected:
        if rates.gpu_usd_per_hour is None:
            missing = True
        else:
            expected_cost += gpu_hours.expected * rates.gpu_usd_per_hour
    if cpu_hours.expected:
        if rates.cpu_usd_per_hour is None:
            missing = True
        else:
            expected_cost += cpu_hours.expected * rates.cpu_usd_per_hour
    cost = None if missing else _range(expected_cost, unit="USD", method="user-supplied rates")

    return WorkEstimate(
        mode=config.mode,
        expected_trials=trials,
        expected_pruned_trials=pruned,
        expected_promoted_trials=promoted,
        expected_full_evaluations=full_evaluations,
        expected_epochs=total_epochs,
        expected_steps=total_steps,
        expected_examples=total_examples,
        duration_seconds=duration,
        gpu_hours=gpu_hours,
        cpu_hours=cpu_hours,
        flops=flops,
        peak_memory_mb=peak_memory or None,
        storage_mb=storage or None,
        monetary_cost_usd=cost,
        assumptions=assumptions + (["monetary cost unavailable until required rates are provided"] if missing else []),
    )

def _session_capacity(
    limit: float | None,
    per_session: float,
) -> int | None:
    """Return how many bounded sessions fit inside a limit."""

    if limit is None or per_session <= 0:
        return None

    if limit <= 0:
        return 0

    # Continuous stopping is evaluated between sessions.
    # Any positive budget therefore permits at least one session,
    # which may overshoot the limit by up to one session.
    return max(
        1,
        math.floor(limit / per_session),
    )


def estimate_continuous_capacity(
    config: StudyConfig,
    *,
    train_examples: int,
    validation_examples: int,
    calibration_records: Sequence[Mapping[str, Any]],
    representative_batch_size: int | None = None,
) -> dict[str, Any]:
    """
    Estimate how much continuous-search work fits inside enabled budgets.

    This is a capacity projection, not a fixed completion estimate.
    Metric targets and stagnation rules cannot be predicted reliably
    before the search begins.
    """

    if not config.continuous.enabled:
        raise ValueError(
            "continuous capacity estimation requires "
            "config.continuous.enabled=True"
        )

    session_config = copy.deepcopy(config)
    session_config.continuous.enabled = False

    if config.mode == "proxy":
        session_config.proxy.trials = 1
        session_config.proxy.promote_top_k = 1
        candidates_per_session = 1

    elif config.mode == "full":
        session_config.full.trials = 1
        candidates_per_session = 1

    else:
        # run_continuous() launches one complete successive-halving
        # session using proxy.trials initial candidates.
        candidates_per_session = config.proxy.trials

    session_estimate = estimate_search(
        session_config,
        train_examples=train_examples,
        validation_examples=validation_examples,
        calibration_records=calibration_records,
        representative_batch_size=representative_batch_size,
    )

    candidate_caps: dict[str, dict[str, int | None]] = {}

    def add_range_cap(
        name: str,
        limit: float | None,
        estimate_range: EstimateRange,
    ) -> None:
        if limit is None:
            return

        candidate_caps[name] = {
            "optimistic": (
                None
                if (
                    capacity := _session_capacity(
                        limit,
                        estimate_range.optimistic,
                    )
                )
                is None
                else capacity * candidates_per_session
            ),
            "expected": (
                None
                if (
                    capacity := _session_capacity(
                        limit,
                        estimate_range.expected,
                    )
                )
                is None
                else capacity * candidates_per_session
            ),
            "conservative": (
                None
                if (
                    capacity := _session_capacity(
                        limit,
                        estimate_range.conservative,
                    )
                )
                is None
                else capacity * candidates_per_session
            ),
        }

    continuous = config.continuous

    add_range_cap(
        "maximum_wall_time_hours",
        (
            None
            if continuous.maximum_wall_time_hours is None
            else continuous.maximum_wall_time_hours * 3600
        ),
        session_estimate.duration_seconds,
    )

    add_range_cap(
        "maximum_gpu_hours",
        continuous.maximum_gpu_hours,
        session_estimate.gpu_hours,
    )

    add_range_cap(
        "maximum_cpu_hours",
        continuous.maximum_cpu_hours,
        session_estimate.cpu_hours,
    )

    if continuous.maximum_cost_usd is not None:
        if session_estimate.monetary_cost_usd is None:
            candidate_caps[
                "maximum_cost_usd"
            ] = {
                "optimistic": None,
                "expected": None,
                "conservative": None,
            }
        else:
            add_range_cap(
                "maximum_cost_usd",
                continuous.maximum_cost_usd,
                session_estimate.monetary_cost_usd,
            )

    if continuous.maximum_trials is not None:
        candidate_caps["maximum_trials"] = {
            "optimistic": continuous.maximum_trials,
            "expected": continuous.maximum_trials,
            "conservative": continuous.maximum_trials,
        }

    scenarios = (
        "optimistic",
        "expected",
        "conservative",
    )

    candidate_capacity: dict[str, int | None] = {}

    for scenario in scenarios:
        available = [
            values[scenario]
            for values in candidate_caps.values()
            if values[scenario] is not None
        ]

        candidate_capacity[scenario] = (
            min(available)
            if available
            else None
        )

    session_capacity = {
        scenario: (
            None
            if candidate_capacity[scenario] is None
            else math.ceil(
                candidate_capacity[scenario]
                / candidates_per_session
            )
        )
        for scenario in scenarios
    }

    expected_capacity = candidate_capacity["expected"]

    limiting_conditions = []

    if expected_capacity is not None:
        limiting_conditions = [
            name
            for name, values in candidate_caps.items()
            if values["expected"] == expected_capacity
        ]

    heuristic_stopping_conditions = []

    if continuous.target_validation_metric is not None:
        heuristic_stopping_conditions.append(
            "target_validation_metric"
        )

    if (
        continuous.stop_after_no_improvement_trials
        is not None
    ):
        heuristic_stopping_conditions.append(
            "no_improvement"
        )

    if continuous.pareto_stagnation_trials is not None:
        heuristic_stopping_conditions.append(
            "pareto_stagnation"
        )

    assumptions = [
        "capacity is projected from one measured bounded session",
        (
            "continuous stopping is checked between sessions; "
            "actual resource use may overshoot by one session"
        ),
        (
            "target accuracy and stagnation stopping conditions "
            "cannot be predicted before observations exist"
        ),
        (
            "sampled optimizers, batch sizes, and architectures "
            "can change session duration"
        ),
    ]

    if (
        continuous.maximum_cost_usd is not None
        and session_estimate.monetary_cost_usd is None
    ):
        assumptions.append(
            "cost capacity is unavailable because required "
            "hourly or storage rates are missing"
        )

    return {
        "mode": config.mode,
        "continuous": True,
        "classification": "budget capacity projection",
        "candidates_per_session": candidates_per_session,
        "session_estimate": session_estimate.to_dict(),
        "candidate_capacity": candidate_capacity,
        "session_capacity": session_capacity,
        "capacity_by_limit": candidate_caps,
        "likely_numeric_limiting_conditions": (
            limiting_conditions
        ),
        "heuristic_stopping_conditions": (
            heuristic_stopping_conditions
        ),
        "assumptions": assumptions,
    }

def compare_estimate(predicted_seconds: float, measured_seconds: Iterable[float]) -> dict[str, Any]:
    measured = [float(item) for item in measured_seconds]
    if not measured:
        return {"calibration_sample_count": 0}
    errors = [predicted_seconds - item for item in measured]
    percentages = [abs(error) / item * 100 for error, item in zip(errors, measured, strict=True) if item > 0]
    return {
        "calibration_sample_count": len(measured),
        "predicted_seconds": predicted_seconds,
        "measured_seconds": measured,
        "absolute_errors_seconds": [abs(item) for item in errors],
        "percentage_errors": percentages,
        "median_absolute_percentage_error": statistics.median(percentages) if percentages else None,
        "underestimation_rate": sum(error < 0 for error in errors) / len(errors),
        "overestimation_rate": sum(error > 0 for error in errors) / len(errors),
    }
