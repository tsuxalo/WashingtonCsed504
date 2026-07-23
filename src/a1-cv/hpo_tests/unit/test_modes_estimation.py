from __future__ import annotations

from pathlib import Path

import pytest

from hpo.config import load_study_config
from hpo.estimation import compare_estimate, estimate_search
from hpo.modes import full_resources, halving_rungs, proxy_resource, rung_resource


def _config(tmp_path: Path, mode="successive_halving"):
    return load_study_config({
        "study": {"name": "test", "output_dir": str(tmp_path), "storage_path": str(tmp_path / "study.db")},
        "search": {"mode": mode, "sampler": "random"},
        "dataset": {"name": "synthetic", "train_examples": 100, "validation_examples": 20, "test_examples": 20},
        "model": {"name": "resnet18"},
        "search_space": {"batch_size": {"type": "fixed", "default": 10}, "optimizer": {"type": "fixed", "default": "sgd"}},
        "proxy": {"trials": 9, "epochs": 1, "maximum_steps": 5, "train_fraction": 0.5, "promote_top_k": 6},
        "successive_halving": {"rung_budgets": [2, 5], "reduction_factor": 3, "minimum_trials_per_rung": 1},
        "full": {"trials": 2, "epochs": 10, "seeds": [1, 2]},
    })


def test_mode_resources(tmp_path: Path):
    config = _config(tmp_path)
    proxy = proxy_resource(config)
    assert proxy.target_epochs == 1 and proxy.max_steps == 5 and proxy.data_fraction == 0.5
    rungs = halving_rungs(config, 6)
    assert [(r.entrants, r.survivors) for r in rungs] == [(6, 2), (2, 1)]
    assert rung_resource(config, rungs[1], 42).target_epochs == 5
    assert len(full_resources(config)) == 2


def test_mode_specific_estimates(tmp_path: Path):
    calibration = [{
        "seconds_per_example": 0.01,
        "evaluation_seconds_per_epoch": 0.1,
        "checkpoint_seconds_per_epoch": 0.05,
        "checkpoint_size_mb": 2,
        "peak_memory_mb": 100,
        "batch_size": 10,
        "train_flops_per_image": 1000,
        "cpu_seconds": 1,
        "elapsed_seconds": 2,
    }]
    staged = estimate_search(_config(tmp_path), train_examples=100, validation_examples=20, calibration_records=calibration)
    assert staged.expected_examples > 0
    assert staged.expected_promoted_trials > 0
    assert staged.flops is not None
    full = estimate_search(_config(tmp_path, "full"), train_examples=100, validation_examples=20, calibration_records=calibration)
    assert full.expected_pruned_trials == 0
    assert full.expected_full_evaluations == 2


def test_estimate_comparison():
    report = compare_estimate(10, [8, 12])
    assert report["calibration_sample_count"] == 2
    assert report["median_absolute_percentage_error"] > 0
