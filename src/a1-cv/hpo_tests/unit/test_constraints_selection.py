from __future__ import annotations

import pytest

from hpo.constraints import check_hard_constraints, validate_candidate
from hpo.exceptions import InvalidTrialError
from hpo.schemas import ConstraintSpec, ObjectiveSpec
from hpo.selection import (
    fastest_above_accuracy,
    highest_accuracy_under_budget,
    lowest_memory_above_accuracy,
    pareto_front,
    pareto_knee,
)


def test_optimizer_constraints():
    with pytest.raises(InvalidTrialError, match="only to SGD"):
        validate_candidate({"optimizer": "adamw", "momentum": 0.9}, {"name": "resnet18"}, {"device": "cpu"})
    with pytest.raises(InvalidTrialError, match="Nesterov"):
        validate_candidate({"optimizer": "sgd", "momentum": 0.0, "nesterov": True}, {"name": "resnet18"}, {"device": "cpu"})


def test_vit_architecture_constraints():
    with pytest.raises(InvalidTrialError, match="divisible"):
        validate_candidate({"optimizer": "adamw", "hidden_dim": 385, "heads": 6}, {"name": "vit"}, {"device": "cpu"})
    with pytest.raises(InvalidTrialError, match="divide"):
        validate_candidate({"optimizer": "adamw", "patch_size": 6}, {"name": "vit"}, {"device": "cpu"})


def test_precision_constraint():
    with pytest.raises(InvalidTrialError, match="fp16"):
        validate_candidate({"optimizer": "sgd", "precision": "fp16"}, {"name": "resnet18"}, {"device": "cpu"})


def test_hard_constraints():
    violations = check_hard_constraints(
        {"wall_seconds": 12, "validation_top1": 0.8},
        [ConstraintSpec("wall_seconds", "<=", 10), ConstraintSpec("validation_top1", ">=", 0.75)],
    )
    assert len(violations) == 1
    assert "wall_seconds" in violations[0]


def test_pareto_and_selection_helpers():
    rows = [
        {"candidate_id": "a", "metrics": {"validation_top1": 0.90, "wall_seconds": 20, "peak_gpu_memory_mb": 500}},
        {"candidate_id": "b", "metrics": {"validation_top1": 0.89, "wall_seconds": 10, "peak_gpu_memory_mb": 300}},
        {"candidate_id": "c", "metrics": {"validation_top1": 0.85, "wall_seconds": 30, "peak_gpu_memory_mb": 600}},
    ]
    objectives = [ObjectiveSpec("validation_top1", "maximize", True), ObjectiveSpec("wall_seconds", "minimize")]
    metric_rows = [row["metrics"] | {"candidate_id": row["candidate_id"]} for row in rows]
    assert {row["candidate_id"] for row in pareto_front(metric_rows, objectives)} == {"a", "b"}
    assert highest_accuracy_under_budget(rows, budget_name="wall_seconds", maximum=15)["candidate_id"] == "b"
    assert fastest_above_accuracy(rows, minimum_accuracy=0.88)["candidate_id"] == "b"
    assert lowest_memory_above_accuracy(rows, minimum_accuracy=0.88)["candidate_id"] == "b"
    assert pareto_knee(rows, objectives)["candidate_id"] in {"a", "b"}
