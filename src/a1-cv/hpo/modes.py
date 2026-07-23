from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .objectives import primary_score
from .schemas import ObjectiveSpec, StudyConfig
from .trial_runner import TrialResource


@dataclass(frozen=True)
class Rung:
    index: int
    resource_type: str
    budget: float
    entrants: int
    survivors: int


def proxy_resource(config: StudyConfig, seed: int | None = None) -> TrialResource:
    budget = config.proxy.budget
    return TrialResource(
        stage="proxy",
        target_epochs=int(budget.epochs or 1),
        max_steps=budget.max_steps,
        data_fraction=budget.data_fraction,
        validation_fraction=budget.validation_fraction,
        seed=config.seed if seed is None else seed,
        continue_checkpoint=False,
    )


def halving_rungs(config: StudyConfig, entrants: int) -> list[Rung]:
    active = entrants
    result = []
    for index, budget in enumerate(config.successive_halving.rung_budgets):
        survivors = max(
            config.successive_halving.minimum_trials_per_rung,
            math.ceil(active / config.successive_halving.reduction_factor),
        )
        survivors = min(active, survivors)
        result.append(Rung(index, config.successive_halving.resource_type, float(budget), active, survivors))
        active = survivors
    return result


def rung_resource(config: StudyConfig, rung: Rung, seed: int) -> TrialResource:
    proxy = config.proxy.budget
    if rung.resource_type == "epochs":
        return TrialResource(
            stage="halving",
            target_epochs=max(1, int(rung.budget)),
            data_fraction=1.0,
            validation_fraction=1.0,
            seed=seed,
            continue_checkpoint=config.successive_halving.continue_checkpoints,
        )
    if rung.resource_type == "steps":
        return TrialResource(
            stage="halving",
            target_epochs=max(1, int(config.full.budget.epochs or proxy.epochs or 1)),
            max_steps=max(1, int(rung.budget)),
            data_fraction=1.0,
            validation_fraction=1.0,
            seed=seed,
            continue_checkpoint=config.successive_halving.continue_checkpoints,
        )
    if rung.resource_type == "data_fraction":
        return TrialResource(
            stage="halving",
            target_epochs=max(1, int(proxy.epochs or 1)),
            data_fraction=min(1.0, float(rung.budget)),
            validation_fraction=1.0,
            seed=seed,
            continue_checkpoint=config.successive_halving.continue_checkpoints,
        )
    if rung.resource_type == "seeds":
        return TrialResource(
            stage="halving",
            target_epochs=max(1, int(config.full.budget.epochs or 1)),
            data_fraction=1.0,
            validation_fraction=1.0,
            seed=seed,
            continue_checkpoint=False,
        )
    raise ValueError(rung.resource_type)


def full_resources(config: StudyConfig) -> list[TrialResource]:
    budget = config.full.budget
    return [
        TrialResource(
            stage="full",
            target_epochs=max(1, int(budget.epochs or 1)),
            max_steps=budget.max_steps,
            data_fraction=budget.data_fraction,
            validation_fraction=budget.validation_fraction,
            seed=seed,
            evaluate_test=config.full.evaluate_test,
            continue_checkpoint=False,
        )
        for seed in budget.seeds
    ]


def rank_candidates(rows: Sequence[Mapping[str, Any]], objective: ObjectiveSpec) -> list[Mapping[str, Any]]:
    valid = [row for row in rows if row.get("status") == "completed" and row.get("metrics")]
    return sorted(valid, key=lambda row: primary_score(row["metrics"], objective), reverse=True)


def promote(rows: Sequence[Mapping[str, Any]], survivors: int, objective: ObjectiveSpec):
    ranked = rank_candidates(rows, objective)
    return ranked[:survivors], ranked[survivors:]
