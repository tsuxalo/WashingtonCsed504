from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpo.conditions import SafeCondition
from hpo.exceptions import SearchSpaceError
from hpo.search_space import (
    combination_count,
    enumerate_combinations,
    load_csv,
    load_space,
    normalize_space,
)


def test_dictionary_and_list_normalize_equivalently():
    dictionary = {
        "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        "batch_size": {"type": "categorical", "choices": [64, 128]},
    }
    as_list = [
        {"name": "learning_rate", "type": "float", "low": 1e-4, "high": 1e-2, "log": True},
        {"name": "batch_size", "type": "categorical", "choices": [64, 128]},
    ]
    assert [item.to_dict() for item in normalize_space(dictionary)] == [
        item.to_dict() for item in normalize_space(as_list)
    ]


def test_csv_conditions_and_typed_choices(tmp_path: Path):
    source = tmp_path / "space.csv"
    source.write_text(
        "name,type,low,high,choices,step,log,default,condition,enabled\n"
        "optimizer,categorical,,,sgd|adamw,,false,sgd,,true\n"
        "momentum,float,0.8,0.9,,0.1,false,0.9,optimizer == 'sgd',true\n",
        encoding="utf-8",
    )
    specs = load_csv(source)
    assert specs[0].choices == ("sgd", "adamw")
    assert specs[1].condition == "optimizer == 'sgd'"


def test_json_and_yaml_loaders(tmp_path: Path):
    value = [{"name": "x", "type": "int", "low": 1, "high": 3}]
    json_path = tmp_path / "space.json"
    json_path.write_text(json.dumps(value), encoding="utf-8")
    yaml_path = tmp_path / "space.yaml"
    yaml_path.write_text("- name: x\n  type: int\n  low: 1\n  high: 3\n", encoding="utf-8")
    assert load_space(json_path)[0].name == "x"
    assert load_space(yaml_path)[0].name == "x"


def test_safe_condition_blocks_calls_and_attributes():
    with pytest.raises(SearchSpaceError, match="unsupported syntax"):
        SafeCondition("__import__('os').system('echo nope')")
    with pytest.raises(SearchSpaceError, match="unsupported syntax"):
        SafeCondition("optimizer.__class__")


def test_unknown_and_cyclic_conditions_are_actionable():
    with pytest.raises(SearchSpaceError, match="unknown parameter"):
        normalize_space([
            {"name": "x", "type": "bool"},
            {"name": "y", "type": "bool", "condition": "missing == True"},
        ])
    with pytest.raises(SearchSpaceError, match="cyclic"):
        normalize_space([
            {"name": "x", "type": "bool", "condition": "y == True"},
            {"name": "y", "type": "bool", "condition": "x == True"},
        ])


@pytest.mark.parametrize(
    "space,pattern",
    [
        ([{"name": "x", "type": "categorical", "choices": []}], "cannot be empty"),
        ([{"name": "x", "type": "float", "low": 1, "high": 0}], "exceeds"),
        ([{"name": "x", "type": "float", "low": 0, "high": 1, "log": True}], "positive"),
        ([{"name": "x", "type": "int", "low": 1, "high": 3, "default": 4}], "outside"),
    ],
)
def test_validation_errors(space, pattern):
    with pytest.raises(SearchSpaceError, match=pattern):
        normalize_space(space)


def test_conditional_combination_count_and_enumeration():
    specs = normalize_space([
        {"name": "optimizer", "type": "categorical", "choices": ["sgd", "adamw"]},
        {"name": "momentum", "type": "categorical", "choices": [0.8, 0.9], "condition": "optimizer == 'sgd'"},
        {"name": "beta1", "type": "categorical", "choices": [0.85, 0.9], "condition": "optimizer == 'adamw'"},
    ])
    combinations = enumerate_combinations(specs, limit=10)
    assert combination_count(specs) == 4
    assert len(combinations) == 4
    assert all(not ({"momentum", "beta1"} <= set(row)) for row in combinations)


def test_continuous_space_has_no_finite_count():
    specs = normalize_space({"learning_rate": {"type": "float", "low": 1e-5, "high": 1e-1, "log": True}})
    assert combination_count(specs) is None
    with pytest.raises(SearchSpaceError, match="fully discrete"):
        enumerate_combinations(specs, limit=100)
