from __future__ import annotations

import time
from pathlib import Path

import pytest

from hpo.config import load_study_config
from hpo.persistence import append_jsonl
from hpo.study import HpoStudy


def _study(tmp_path: Path, continuous: dict) -> HpoStudy:
    config = load_study_config({
        "study": {"name": "continuous-conditions", "output_dir": str(tmp_path), "storage_path": str(tmp_path / "study.db")},
        "search": {"mode": "proxy", "sampler": "random"},
        "dataset": {"name": "synthetic", "train_examples": 8, "validation_examples": 4, "test_examples": 4, "num_classes": 10},
        "model": {"name": "vit", "hidden_dim": 48, "layers": 1, "heads": 3, "mlp_dim": 96, "patch_size": 8},
        "search_space": {"batch_size": {"type": "fixed", "default": 4}, "optimizer": {"type": "fixed", "default": "adamw"}, "learning_rate": {"type": "fixed", "default": 0.001}},
        "runtime": {"device": "cpu", "precision": "fp32", "concurrent_trials": 1, "intraop_threads": 1, "interop_threads": 1},
        "proxy": {"trials": 1, "epochs": 1, "maximum_steps": 1, "promote_top_k": 1},
        "continuous": {"enabled": True, "strategy": "proxy", **continuous},
    })
    # repo root is three parents above src/a1-cv in the test copy
    return HpoStudy(config, repo_root=Path(__file__).resolve().parents[4])


@pytest.mark.parametrize(
    "continuous,started,best,no_improvement,stagnation,state_hashes,metrics",
    [
        ({"maximum_trials": 1}, None, 0.0, 0, 0, ["x"], {}),
        ({"maximum_wall_time_hours": 0.00001}, -1.0, 0.0, 0, 0, [], {}),
        ({"maximum_gpu_hours": 0.5}, None, 0.0, 0, 0, [], {"gpu_hours": 0.5}),
        ({"maximum_cpu_hours": 0.5}, None, 0.0, 0, 0, [], {"cpu_hours": 0.5}),
        ({"maximum_cost_usd": 1.0}, None, 0.0, 0, 0, [], {"known_component_total_usd": 1.0}),
        ({"target_validation_metric": 0.9}, None, 0.9, 0, 0, [], {}),
        ({"stop_after_no_improvement_trials": 2}, None, 0.0, 2, 0, [], {}),
        ({"pareto_stagnation_trials": 2}, None, 0.0, 0, 2, [], {}),
    ],
)
def test_every_continuous_stop_condition(
    tmp_path, continuous, started, best, no_improvement, stagnation, state_hashes, metrics
):
    study = _study(tmp_path, continuous)
    study.state["completed_param_hashes"] = state_hashes
    if metrics:
        append_jsonl(study.files.results, {"status": "completed", "metrics": metrics})
    start = time.time() if started is None else time.time() + started * 3600
    assert study._continuous_stop(start, best, no_improvement, stagnation) is True
