from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import torch

CV_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = CV_DIR.parents[1]
for path in (CV_DIR, REPO_ROOT / "src" / "common"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hpo.adapters import RepoModules, build_trial_dataset, build_trial_model
from hpo.config import load_study_config
from hpo.persistence import read_jsonl
from hpo.study import HpoStudy


def base_mapping(root: Path, mode: str) -> dict:
    output = root / mode
    return {
        "study": {
            "name": f"tiny-{mode}",
            "output_dir": str(output),
            "storage_path": str(output / f"tiny-{mode}.db"),
            "seed": 7,
            "resume": True,
        },
        "search": {"mode": mode, "sampler": "random"},
        "dataset": {
            "name": "synthetic", "train_examples": 16,
            "validation_examples": 16, "test_examples": 16,
            "num_classes": 10,
        },
        "model": {
            "name": "vit", "hidden_dim": 96, "layers": 1,
            "heads": 3, "mlp_dim": 192, "patch_size": 8,
        },
        "search_space": {
            "batch_size": {"type": "fixed", "default": 16},
            "optimizer": {"type": "fixed", "default": "adamw"},
            "learning_rate": {"type": "categorical", "choices": [0.0005, 0.001], "default": 0.001},
            "weight_decay": {"type": "fixed", "default": 0.05},
            "beta1": {"type": "fixed", "default": 0.9},
            "beta2": {"type": "fixed", "default": 0.999},
            "scheduler": {"type": "fixed", "default": "cosine"},
            "warmup_epochs": {"type": "fixed", "default": 0},
            "label_smoothing": {"type": "fixed", "default": 0.0},
            "gradient_clip": {"type": "fixed", "default": 1.0},
            "strong_augmentation": {"type": "fixed", "default": False},
            "channels_last": {"type": "fixed", "default": False},
        },
        "runtime": {
            "device": "cpu", "precision": "fp32", "concurrent_trials": 1,
            "intraop_threads": 2, "interop_threads": 1,
        },
        "proxy": {
            "trials": 2, "epochs": 1, "maximum_steps": 1,
            "train_fraction": 1.0, "validation_fraction": 1.0,
            "promote_top_k": 1,
        },
        "successive_halving": {
            "resource_type": "epochs", "rung_budgets": [2],
            "reduction_factor": 2, "minimum_trials_per_rung": 1,
            "continue_checkpoints": True,
        },
        "full": {
            "enabled": True, "trials": 1, "epochs": 1,
            "maximum_steps": 1, "seeds": [7], "evaluate_test": False,
        },
        "continuous": {"enabled": False},
        "objectives": [
            {"name": "validation_top1", "direction": "maximize", "primary": True},
            {"name": "wall_seconds", "direction": "minimize"},
        ],
    }


def run() -> dict:
    root = Path(tempfile.mkdtemp(prefix="washington-hpo-integration-"))
    results = {}
    try:
        for mode in ("proxy", "successive_halving", "full"):
            config = load_study_config(base_mapping(root, mode))
            summary = HpoStudy(config, repo_root=REPO_ROOT).run()
            study_dir = config.output_dir / config.name
            assert summary["mode"] == mode
            assert summary["records"] >= 1
            assert (study_dir / "all_trials.csv").exists()
            assert (study_dir / "pareto_trials.csv").exists()
            results[mode] = summary

        mapping = base_mapping(root, "proxy")
        mapping["study"]["name"] = "tiny-continuous"
        mapping["study"]["storage_path"] = str(root / "proxy" / "tiny-continuous.db")
        mapping["continuous"] = {
            "enabled": True,
            "strategy": "proxy",
            "maximum_trials": 2,
            "maximum_wall_time_hours": 0.1,
            "stop_after_no_improvement_trials": None,
            "pareto_stagnation_trials": None,
        }
        continuous = load_study_config(mapping)
        first = HpoStudy(continuous, repo_root=REPO_ROOT).run()
        study_dir = continuous.output_dir / continuous.name
        rows_before = read_jsonl(study_dir / "trials.jsonl")
        hashes_before = json.loads((study_dir / "state.json").read_text())["completed_param_hashes"]
        assert len(set(hashes_before)) == 2
        second = HpoStudy(continuous, repo_root=REPO_ROOT).run()
        assert len(read_jsonl(study_dir / "trials.jsonl")) == len(rows_before)
        assert json.loads((study_dir / "state.json").read_text())["completed_param_hashes"] == hashes_before
        results["continuous"] = second

        config = load_study_config(base_mapping(root, "full"))
        # Use a distinct study name for the explicit resume check.
        config.name = "tiny-resume"
        config.storage_path = config.output_dir / "tiny-resume.db"
        first = HpoStudy(config, repo_root=REPO_ROOT).run()
        rows_before = read_jsonl(config.output_dir / config.name / "trials.jsonl")
        second = HpoStudy(config, repo_root=REPO_ROOT).run()
        rows_after = read_jsonl(config.output_dir / config.name / "trials.jsonl")
        assert first == second
        assert len(rows_before) == len(rows_after)
        results["resume"] = second

        modules = RepoModules(REPO_ROOT)
        device = torch.device("cpu")
        dataset = build_trial_dataset(
            modules,
            {"name": "synthetic", "train_examples": 8, "validation_examples": 4, "test_examples": 4, "num_classes": 10},
            {"seed": 1},
            device,
        )
        x, _ = next(dataset.train.epoch(4, False))
        resnet = build_trial_model(modules, {"name": "resnet18"}, 10, device)
        vit = build_trial_model(
            modules,
            {"name": "vit", "hidden_dim": 96, "layers": 1, "heads": 3, "mlp_dim": 192, "patch_size": 8},
            10,
            device,
        )
        assert resnet(x).shape == (4, 10)
        assert vit(x).shape == (4, 10)
        results["adapters"] = "passed"
        return results
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
